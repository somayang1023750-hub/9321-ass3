"""Train movie revenue/rating models and write the two assignment CSV files.

Usage: python3 z5429763.py <train_file> <test_file>
Only the training file is used for fitting; test targets are never read as features.
"""

from __future__ import annotations

import argparse
from collections import Counter
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor

SEED = 42
THREADS = 4
# Both targets, IDs, and constant status are excluded from all predictors.
# vote_average is used only as the numeric input supplied by the specification;
# it is never calculated from rating or revenue.
EXCLUDED_COLUMNS = ("revenue", "rating", "imdb_id", "status")
NUMERIC_COLUMNS = (
    "budget",
    "runtime",
    "vote_count",
    "imdb_votes",
    "popularity",
    "vote_average",
)
LIST_COLUMNS = (
    "genres",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "cast",
    "writers",
    "producers",
    "keywords",
)
CATEGORY_COLUMNS = (
    "original_language",
    "director",
    "director_of_photography",
    "music_composer",
)
GENRES = (
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
)
REGRESSION_PARAMS = dict(iterations=650, depth=6, learning_rate=0.06, l2_leaf_reg=5)
CLASSIFICATION_PARAMS = dict(iterations=650, depth=6, learning_rate=0.06, l2_leaf_reg=5)
# The original regressor factory remains available for notebook comparisons.
REVENUE_CAT_PARAMS = dict(
    iterations=1200,
    depth=7,
    learning_rate=0.04,
    l2_leaf_reg=10,
    max_ctr_complexity=1,
)
REVENUE_LGB_PARAMS = dict(
    n_estimators=276,
    num_leaves=63,
    learning_rate=0.025,
    min_child_samples=20,
    reg_lambda=20,
    colsample_bytree=0.9,
)


def text_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a normalized string column, retaining explicit missing values."""
    return df[name].fillna("").astype(str).str.strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-wise features: no fit, target access, or train/test joins.

    CatBoost learns all category encodings exclusively during model.fit on the
    training partition. Missing/unseen categories at prediction are supported.
    """
    features = pd.DataFrame(index=df.index)
    for name in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[name], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        values = values.mask(values < 0)
        if name in ("budget", "runtime"):
            values = values.mask(values == 0)  # zero means unknown, not a free film
        features[name] = values
        features[f"{name}_missing"] = values.isna().astype("int8")
        if name != "runtime":
            features[f"log1p_{name}"] = np.log1p(values)
    dates = pd.to_datetime(df["release_date"], errors="coerce", format="mixed")
    features["release_year"] = dates.dt.year.astype(float)
    features["release_month"] = dates.dt.month.astype(float)
    features["release_quarter"] = dates.dt.quarter.astype(float)
    features["release_date_missing"] = dates.isna().astype("int8")
    for name in CATEGORY_COLUMNS:
        features[name] = text_column(df, name).replace("", "__MISSING__")
    for name in LIST_COLUMNS:
        values = text_column(df, name)
        separator = "|" if name == "keywords" else ","
        tokens = values.map(
            lambda value: [t.strip() for t in value.split(separator) if t.strip()]
        )
        features[f"{name}_count"] = tokens.map(len).astype(float)
        features[f"{name}_first"] = tokens.map(
            lambda ts: ts[0] if ts else "__MISSING__"
        )
    genres = text_column(df, "genres").map(lambda s: {v.strip() for v in s.split(",")})
    for genre in GENRES:
        features[f"genre_{genre}"] = genres.map(lambda gs: int(genre in gs)).astype(
            "int8"
        )
    for name in ("title", "original_title", "overview", "tagline"):
        values = text_column(df, name)
        features[f"{name}_length"] = values.str.len().astype(float)
        features[f"{name}_words"] = values.str.split().str.len().astype(float)
        features[f"{name}_missing"] = values.eq("").astype("int8")
    features["title_is_original"] = (
        text_column(df, "title").eq(text_column(df, "original_title")).astype("int8")
    )
    return features


def regression_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise credits and ratios; neither targets nor test statistics are used."""
    f = preprocess(df)
    for column, sep, count in [
        ("cast", ",", 4),
        ("production_companies", ",", 3),
        ("production_countries", ",", 3),
        ("spoken_languages", ",", 3),
        ("writers", ",", 3),
        ("producers", ",", 3),
        ("keywords", "|", 4),
        ("genres", ",", 3),
    ]:
        tokens = text_column(df, column).map(
            lambda t: [q.strip() for q in t.split(sep) if q.strip()]
        )
        for pos in range(1, count):
            f[f"{column}_{pos + 1}"] = tokens.map(
                lambda v: v[pos] if len(v) > pos else "__MISSING__"
            )
        if column in (
            "production_companies",
            "production_countries",
            "genres",
            "spoken_languages",
        ):
            f[column + "_all"] = tokens.map(
                lambda v: " | ".join(sorted(v)) if v else "__MISSING__"
            )
    f["release_decade"] = f.release_year // 10 * 10
    for numerator, denominator in [
        ("imdb_votes", "vote_count"),
        ("popularity", "vote_count"),
        ("budget", "runtime"),
    ]:
        f[numerator + "_per_" + denominator] = np.log1p(f[numerator]) - np.log1p(
            f[denominator]
        )
    for col in ["budget", "vote_count", "imdb_votes", "popularity"]:
        f[col + "_small"] = f[col].between(1, 1000).astype("int8")
    return f


class EntityFrequencyFeatures:
    """Fit-only, role-specific counts of distinct entities per film.

    During fitting, leave-one-out counts exclude the current film. Prediction
    uses the complete fitted dictionary and never learns from the input batch.
    """

    columns = (
        "cast",
        "production_companies",
        "director",
        "writers",
        "producers",
        "keywords",
    )

    @staticmethod
    def _tokens(df, name):
        separator = "|" if name == "keywords" else ","
        return text_column(df, name).map(
            lambda value: list(
                dict.fromkeys(
                    token.strip().casefold()
                    for token in value.split(separator)
                    if token.strip()
                )
            )
        )

    def fit(self, df):
        self.counts_ = {}
        for name in self.columns:
            self.counts_[name] = Counter(
                token for tokens in self._tokens(df, name) for token in tokens
            )
        return self

    def transform(self, df, leave_one_out=False):
        features = pd.DataFrame(index=df.index)
        subtract = int(leave_one_out)
        for name in self.columns:
            counts = self.counts_[name]
            values = self._tokens(df, name).map(
                lambda tokens: np.array(
                    [max(0, counts.get(token, 0) - subtract) for token in tokens],
                    dtype=float,
                )
            )
            features[name + "_all_freq_max"] = values.map(
                lambda values: np.log1p(values.max()) if len(values) else 0.0
            ).to_numpy()
            features[name + "_all_freq_mean"] = values.map(
                lambda values: np.log1p(values).mean() if len(values) else 0.0
            ).to_numpy()
            features[name + "_all_freq_known"] = values.map(
                lambda values: np.mean(values > 0) if len(values) else 0.0
            ).to_numpy()
        return features


def add_revenue_crosses(features):
    """Expose three low-cardinality conditional categories, without targets."""
    features = features.copy()
    state = features.budget_missing.astype(str)
    features["country_budget_state"] = (
        features.production_countries_first + " | " + state
    )
    features["genre_budget_state"] = features.genres_first + " | " + state
    decade = features.release_decade.fillna(-1).astype(int).astype(str)
    features["language_decade"] = features.original_language + " | " + decade
    return features


class ReferenceRevenueRegressor:
    """Previous log-space ensemble, retained for reproducible comparisons.

    Category vocabularies are learned only in fit. No validation/test labels or
    target-derived input features are used.
    """

    def _features(self, df, fitting=False):
        return regression_preprocess(df)

    def fit(self, df: pd.DataFrame, log_revenue: np.ndarray):
        features = self._features(df, fitting=True)
        self.cat_model = CatBoostRegressor(
            **REVENUE_CAT_PARAMS,
            loss_function="RMSE",
            random_seed=SEED,
            thread_count=THREADS,
            verbose=False,
            allow_writing_files=False,
        )
        self.cat_model.fit(
            features, log_revenue, cat_features=categorical_columns(features)
        )
        self.lgb_model = LGBMRegressor(
            **REVENUE_LGB_PARAMS,
            n_jobs=THREADS,
            random_state=SEED,
            verbosity=-1,
        )
        # LightGBM requires JSON-safe column names. Keep the mapping fixed.
        self.feature_names = [
            re.sub(r"[^A-Za-z0-9_]+", "_", name) for name in features.columns
        ]
        self.category_dtypes = {}
        light_features = features.copy()
        light_features.columns = self.feature_names
        for name in categorical_columns(light_features):
            dtype = pd.CategoricalDtype(categories=pd.unique(light_features[name]))
            self.category_dtypes[name] = dtype
            light_features[name] = light_features[name].astype(dtype)
        self.lgb_model.fit(light_features, log_revenue)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        features = self._features(df)
        cat_prediction = self.cat_model.predict(features)
        light_features = features.copy()
        light_features.columns = self.feature_names
        for name, dtype in self.category_dtypes.items():
            # Explicitly map previously unseen values to missing, using fit-only vocabularies.
            light_features[name] = (
                light_features[name]
                .where(light_features[name].isin(dtype.categories), None)
                .astype(dtype)
            )
        return 0.5 * cat_prediction + 0.5 * self.lgb_model.predict(light_features)


class RevenueRegressor(ReferenceRevenueRegressor):
    """Conditional categorical crosses plus fit-only leave-one-out entity counts."""

    def _features(self, df, fitting=False):
        features = add_revenue_crosses(regression_preprocess(df))
        if fitting:
            self.entity_frequency_ = EntityFrequencyFeatures().fit(df)
        frequency = self.entity_frequency_.transform(df, leave_one_out=fitting)
        for name in frequency:
            features[name] = frequency[name].to_numpy()
        return features


def categorical_columns(features: pd.DataFrame) -> list[str]:
    return features.select_dtypes(include=["object", "string"]).columns.tolist()


def make_regressor(**overrides) -> CatBoostRegressor:
    params = dict(REGRESSION_PARAMS)
    params.update(overrides)
    return CatBoostRegressor(
        **params,
        loss_function="RMSE",
        random_seed=SEED,
        thread_count=THREADS,
        verbose=False,
        allow_writing_files=False,
    )


def make_classifier(**overrides) -> CatBoostClassifier:
    params = dict(CLASSIFICATION_PARAMS)
    params.update(overrides)
    return CatBoostClassifier(
        **params,
        loss_function="MultiClass",
        auto_class_weights="SqrtBalanced",
        random_seed=SEED,
        thread_count=THREADS,
        verbose=False,
        allow_writing_files=False,
    )


def revenue_from_log(predictions: np.ndarray) -> np.ndarray:
    # Float overflow protection only; no clipping based on test target values.
    return np.maximum(0.0, np.expm1(np.clip(predictions, 0.0, 50.0)))


def train_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    train_features = preprocess(train_df)
    test_features = preprocess(test_df)
    categories = categorical_columns(train_features)
    regressor = RevenueRegressor()
    classifier = make_classifier()
    regressor.fit(train_df, np.log1p(train_df["revenue"]).to_numpy())
    classifier.fit(train_features, train_df["rating"], cat_features=categories)
    revenue = revenue_from_log(regressor.predict(test_df))
    rating = classifier.predict(test_features).reshape(-1)
    return revenue, rating


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_file", help="CSV used for training, including both targets"
    )
    parser.add_argument("test_file", help="CSV to predict; target columns are ignored")
    args = parser.parse_args()
    started = perf_counter()
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
    revenue, rating = train_predict(train_df, test_df)
    prefix = Path(__file__).stem
    pd.DataFrame({"imdb_id": test_df["imdb_id"].to_numpy(), "revenue": revenue}).to_csv(
        f"{prefix}_regression.csv",
        index=False,
    )
    pd.DataFrame({"imdb_id": test_df["imdb_id"].to_numpy(), "rating": rating}).to_csv(
        f"{prefix}_classification.csv",
        index=False,
    )
    print(
        f"Wrote {len(test_df)} predictions per task in {perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
