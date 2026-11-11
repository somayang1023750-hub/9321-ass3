"""Training-only ablations of the uploaded regression proposal.

Only the caller-supplied training CSV is opened. Default runs are development
experiments, not the two-minute submission command. No model/prediction changes
are made by this tool.
"""

import argparse
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import z5429763 as base

FREQUENCY_COLUMNS = (
    "director",
    "production_companies_first",
    "cast_first",
    "original_language",
)


def feature_frame(df, groups=()):
    f = base.regression_preprocess(df)
    if "missing" in groups:
        f["log1p_budget_clean"] = f.log1p_budget.fillna(-1.0)
        f["log1p_runtime_clean"] = np.log1p(f.runtime).fillna(-1.0)
        ratio_missing = f.budget.isna() | f.runtime.isna()
        f["budget_per_runtime_missing"] = ratio_missing.astype("int8")
        f["budget_per_runtime"] = (f.log1p_budget - np.log1p(f.runtime)).mask(
            ratio_missing, -99.0
        )
        f["votes_x_popularity"] = f.log1p_imdb_votes * f.log1p_popularity
        f["unknown_budget_vote_proxy"] = f.budget_missing * f.log1p_imdb_votes
        f["known_budget_vote_ratio"] = (1 - f.budget_missing) * (
            f.log1p_budget.fillna(0) - f.log1p_imdb_votes.fillna(0)
        )
        # imdb_votes_per_vote_count already equals the suggested vote_diff.
        f["is_tmdb_zero_vote"] = f.vote_count.eq(0).astype("int8")
        if "literal" in groups:
            f["budget_per_runtime"] = np.where(
                f.budget_missing.eq(0), f.log1p_budget - np.log1p(f.runtime), -99.0
            )
            f = f.drop(columns=["budget_per_runtime_missing"])
            f["vote_diff"] = f.log1p_imdb_votes - f.log1p_vote_count
    if "release" in groups:
        crew = [
            "director",
            "director_of_photography",
            "music_composer",
            "writers_first",
            "producers_first",
        ]
        f["crew_completeness"] = sum(
            f[col].ne("__MISSING__").astype("int8") for col in crew
        ) + (1 - f.tagline_missing)
        f["is_international_release"] = (
            f.spoken_languages_count.gt(1) | f.production_countries_count.gt(1)
        ).astype("int8")
        f["non_en_with_translated_title"] = (
            f.original_language.ne("en") & f.title_is_original.eq(0)
        ).astype("int8")
        words = base.text_column(df, "keywords").str.lower()
        f["is_short_or_indie_kw"] = (
            words.str.contains(
                r"short film|independent film|student film|stand-up|concert", regex=True
            )
            | f.runtime.lt(65)
        ).astype("int8")
    return f


def add_frequency(first, second, include_missing=False):
    first = first.copy()
    second = second.copy()
    maps = {}
    for name in FREQUENCY_COLUMNS:
        # Missing metadata is not an entity's industry activity.
        counts = (
            first[name].value_counts()
            if include_missing
            else first.loc[first[name].ne("__MISSING__"), name].value_counts()
        )
        maps[name] = counts
        for frame in [first, second]:
            frame[name + "_freq"] = frame[name].map(counts).fillna(0).astype(float)
    return first, second, maps


# Freeze the reference, so these experiments remain reproducible after tuning.
REFERENCE_CAT = dict(
    iterations=1200, depth=7, learning_rate=0.04, l2_leaf_reg=10, max_ctr_complexity=1
)
REFERENCE_LIGHT = dict(
    n_estimators=276,
    num_leaves=63,
    learning_rate=0.025,
    min_child_samples=20,
    reg_lambda=20,
    colsample_bytree=0.9,
)
PROPOSED_CAT = dict(
    iterations=1100, depth=7, learning_rate=0.04, l2_leaf_reg=10, max_ctr_complexity=2
)
PROPOSED_LIGHT = dict(
    n_estimators=450,
    num_leaves=45,
    learning_rate=0.025,
    min_child_samples=25,
    reg_lambda=25.0,
    reg_alpha=5.0,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    cat_smooth=35.0,
    cat_l2=25.0,
    min_data_per_group=50,
)
VARIANTS = {
    "reference": ((), False, False, 1),
    "missing": (("missing",), False, False, 1),
    "frequency": ((), True, False, 1),
    "release": (("release",), False, False, 1),
    "frequency_release": (("release",), True, False, 1),
    "all_features": (("missing", "release"), True, False, 1),
    "regularized_only": ((), False, True, 1),
    "frequency_regularized": ((), True, True, 1),
    "all_regularized_ctr1": (("missing", "release"), True, True, 1),
    "all_regularized_ctr2": (("missing", "release"), True, True, 2),
    "literal_full_ctr1": (("missing", "release", "literal"), "literal", True, 1),
    "literal_full_ctr2": (("missing", "release", "literal"), "literal", True, 2),
}


def run_split(train, seed, variants):
    fitting, validation = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=seed, stratify=train.rating
    )
    target = np.log1p(train.revenue.to_numpy())
    cat_cache, light_cache = {}, {}
    results = []
    for name in ["reference"] + [name for name in variants if name != "reference"]:
        groups, frequency, regularized, ctr = VARIANTS[name]
        features = feature_frame(train, groups)
        first, second = features.iloc[fitting].copy(), features.iloc[validation].copy()
        if frequency:
            first, second, _ = add_frequency(
                first, second, include_missing=frequency == "literal"
            )
        cat_params = dict(PROPOSED_CAT if ctr == 2 else REFERENCE_CAT)
        light_params = dict(PROPOSED_LIGHT if regularized else REFERENCE_LIGHT)
        if "literal" in groups:
            cat_params.update(iterations=1100, max_ctr_complexity=ctr)
            # The pasted prose specifies 50, but its executable block omits it.
            light_params.pop("min_data_per_group", None)
        feature_key = (groups, frequency)
        cat_key = (feature_key, tuple(sorted(cat_params.items())))
        light_key = (feature_key, tuple(sorted(light_params.items())))
        if cat_key not in cat_cache:
            started = perf_counter()
            model = CatBoostRegressor(
                **cat_params,
                loss_function="RMSE",
                random_seed=42,
                thread_count=4,
                verbose=False,
                allow_writing_files=False
            )
            model.fit(
                first, target[fitting], cat_features=base.categorical_columns(first)
            )
            cat_cache[cat_key] = (model.predict(second), perf_counter() - started)
        if light_key not in light_cache:
            first_light, second_light = first.copy(), second.copy()
            first_light.columns = [
                re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in first_light.columns
            ]
            second_light.columns = first_light.columns
            for col in base.categorical_columns(first_light):
                dtype = pd.CategoricalDtype(categories=pd.unique(first_light[col]))
                first_light[col] = first_light[col].astype(dtype)
                second_light[col] = (
                    second_light[col]
                    .where(second_light[col].isin(dtype.categories), None)
                    .astype(dtype)
                )
            started = perf_counter()
            model = LGBMRegressor(
                **light_params, n_jobs=4, random_state=42, verbosity=-1
            )
            model.fit(first_light, target[fitting])
            light_cache[light_key] = (
                model.predict(second_light),
                perf_counter() - started,
            )
        cat_prediction, cat_seconds = cat_cache[cat_key]
        light_prediction, light_seconds = light_cache[light_key]
        for weight in ([0.5] if name == "reference" else [0.5, 0.65]):
            prediction = np.maximum(
                0, weight * cat_prediction + (1 - weight) * light_prediction
            )
            error = (target[validation] - prediction) ** 2
            row = dict(
                seed=seed,
                name=name,
                cat_weight=weight,
                RMSLE=float(np.sqrt(error.mean())),
                component_times=dict(
                    cat_seconds=cat_seconds, lightgbm_seconds=light_seconds
                ),
            )
            for group, mask in {
                "budget_unknown": train.budget.iloc[validation].eq(0).to_numpy(),
                "revenue_le_1000": train.revenue.iloc[validation].le(1000).to_numpy(),
                "other": train.revenue.iloc[validation].gt(1000).to_numpy(),
            }.items():
                row[group] = float(np.sqrt(error[mask].mean()))
            results.append(row)
            print(json.dumps(row), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026])
    parser.add_argument(
        "--cases", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    args = parser.parse_args()
    train = pd.read_csv(args.train_file)
    results = []
    for seed in args.seeds:
        results.extend(run_split(train, seed, args.cases))
        Path(args.output).write_text(
            json.dumps(dict(reference_commit="8406404", results=results), indent=2)
        )


if __name__ == "__main__":
    main()
