"""Train movie revenue/rating models and write the two assignment CSV files.

Usage: python3 z5429763.py <train_file> <test_file>
Only the training file is used for fitting; test targets are never read as features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

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
    regressor = make_regressor()
    classifier = make_classifier()
    regressor.fit(
        train_features, np.log1p(train_df["revenue"]), cat_features=categories
    )
    classifier.fit(train_features, train_df["rating"], cat_features=categories)
    revenue = revenue_from_log(regressor.predict(test_features))
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
