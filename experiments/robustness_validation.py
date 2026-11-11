"""Training-only loss/weight/imputation diagnostics; score ALL validation rows.

Usage: python experiments/robustness_validation.py <train_file> --output <report.json>
Use --cases current for only the baseline, group decomposition and budget masking.
The complete development experiment takes longer than the submission CLI.
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
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import z5429763 as solution


def light_frames(fitting, validation):
    fitting, validation = fitting.copy(), validation.copy()
    fitting.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", name) for name in fitting.columns]
    validation.columns = fitting.columns
    for name in solution.categorical_columns(fitting):
        dtype = pd.CategoricalDtype(categories=pd.unique(fitting[name]))
        fitting[name] = fitting[name].astype(dtype)
        validation[name] = (
            validation[name]
            .where(validation[name].isin(dtype.categories), None)
            .astype(dtype)
        )
    return fitting, validation


def fit_pair(fitting, target, validation, weights=None, delta=None):
    cat = CatBoostRegressor(
        **solution.REVENUE_CAT_PARAMS,
        loss_function="RMSE" if delta is None else f"Huber:delta={delta}",
        thread_count=solution.THREADS,
        random_seed=solution.SEED,
        verbose=False,
        allow_writing_files=False,
    )
    cat.fit(
        fitting,
        target,
        cat_features=solution.categorical_columns(fitting),
        sample_weight=weights,
    )
    first, second = light_frames(fitting, validation)
    params = dict(solution.REVENUE_LGB_PARAMS)
    if delta is not None:
        params.update(objective="huber", alpha=0.9)
    light = LGBMRegressor(
        **params, n_jobs=solution.THREADS, random_state=solution.SEED, verbosity=-1
    )
    light.fit(first, target, sample_weight=weights)
    return 0.5 * cat.predict(validation) + 0.5 * light.predict(second), cat, light


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["current", "weights", "huber", "impute", "oof"],
        default=["current", "weights", "huber", "impute", "oof"],
    )
    args = parser.parse_args()
    raw = pd.read_csv(args.train_file)
    target = np.log1p(raw.revenue.to_numpy())
    fitting, validation = train_test_split(
        np.arange(len(raw)), test_size=0.2, random_state=args.seed, stratify=raw.rating
    )
    features = solution.regression_preprocess(raw)
    first, second = features.iloc[fitting].copy(), features.iloc[validation].copy()
    observed = raw.iloc[validation]
    report = dict(
        seed=args.seed,
        fitting_rows=len(fitting),
        validation_rows=len(validation),
        results=[],
    )

    def record(name, prediction, seconds):
        prediction = np.maximum(0, prediction)
        error = (prediction - target[validation]) ** 2
        row = dict(name=name, RMSLE=float(np.sqrt(error.mean())), seconds=seconds)
        for group, mask in {
            "tiny": observed.revenue.le(1000).to_numpy(),
            "other": observed.revenue.gt(1000).to_numpy(),
            "unknown_budget": observed.budget.eq(0).to_numpy(),
            "known_budget": observed.budget.gt(0).to_numpy(),
        }.items():
            row[group] = float(np.sqrt(error[mask].mean()))
        report["results"].append(row)
        print(json.dumps(row), flush=True)

    start = perf_counter()
    baseline, cat, light = fit_pair(first, target[fitting], second)
    record("current", baseline, perf_counter() - start)
    group_data = observed[["revenue", "budget"]].copy()
    group_data["budget_missing"] = group_data.budget.eq(0)
    group_data["tiny_revenue"] = group_data.revenue.le(1000)
    group_data["error"] = (np.maximum(0, baseline) - target[validation]) ** 2
    breakdown = group_data.groupby(["budget_missing", "tiny_revenue"]).error.agg(
        ["size", "mean", "sum"]
    )
    breakdown["RMSLE"] = np.sqrt(breakdown["mean"])
    breakdown["overall_loss_share"] = breakdown["sum"] / group_data.error.sum()
    report["budget_decomposition"] = json.loads(
        breakdown.reset_index().to_json(orient="records")
    )
    masked = observed.copy()
    masked["budget"] = 0
    masked_features = solution.regression_preprocess(masked)
    _, light_masked = light_frames(first, masked_features)
    prediction = 0.5 * cat.predict(masked_features) + 0.5 * light.predict(light_masked)
    known = observed.budget.gt(0).to_numpy()
    report["budget_masking"] = dict(
        known_budget_rows=int(known.sum()),
        observed_budget_RMSLE=float(
            np.sqrt(np.mean((target[validation][known] - baseline[known]) ** 2))
        ),
        masked_budget_RMSLE=float(
            np.sqrt(np.mean((target[validation][known] - prediction[known]) ** 2))
        ),
    )
    if "weights" in args.cases:
        for weight in [0, 0.25, 0.5]:
            start = perf_counter()
            weights = np.where(raw.revenue.iloc[fitting].le(1000), weight, 1.0)
            if weight == 0:
                keep = weights > 0
                prediction, _, _ = fit_pair(
                    first.iloc[keep], target[fitting][keep], second
                )
            else:
                prediction, _, _ = fit_pair(first, target[fitting], second, weights)
            record(f"tiny_weight_{weight}", prediction, perf_counter() - start)
    if "huber" in args.cases:
        for delta in [1.0, 2.0]:
            start = perf_counter()
            prediction, _, _ = fit_pair(first, target[fitting], second, delta=delta)
            record(f"huber_{delta}", prediction, perf_counter() - start)
    if "impute" in args.cases:
        start = perf_counter()
        known = raw.budget.iloc[fitting].gt(0).to_numpy()
        columns = [name for name in first.columns if "budget" not in name]
        estimator = CatBoostRegressor(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=10,
            max_ctr_complexity=1,
            thread_count=solution.THREADS,
            random_seed=solution.SEED,
            verbose=False,
            allow_writing_files=False,
        )
        estimator.fit(
            first.iloc[known][columns],
            np.log1p(raw.budget.iloc[fitting][known]),
            cat_features=solution.categorical_columns(first[columns]),
        )

        def fill(frame, data):
            result = frame.copy()
            missing = data.budget.le(0).to_numpy()
            estimate = np.maximum(0, estimator.predict(frame[columns]))
            result.loc[missing, "budget"] = np.expm1(estimate[missing])
            result.loc[missing, "log1p_budget"] = estimate[missing]
            result["budget_per_runtime"] = np.log1p(result.budget) - np.log1p(
                result.runtime
            )
            result["budget_small"] = result.budget.between(1, 1000).astype("int8")
            return result

        prediction, _, _ = fit_pair(
            fill(first, raw.iloc[fitting]), target[fitting], fill(second, observed)
        )
        record("model_budget_imputation", prediction, perf_counter() - start)
    if "oof" in args.cases:
        start = perf_counter()
        out_of_fold = np.zeros(len(fitting))
        for inner_fit, inner_validation in KFold(
            3, shuffle=True, random_state=42
        ).split(fitting):
            estimator = CatBoostRegressor(
                iterations=400,
                depth=5,
                learning_rate=0.06,
                l2_leaf_reg=10,
                max_ctr_complexity=1,
                thread_count=solution.THREADS,
                random_seed=solution.SEED,
                verbose=False,
                allow_writing_files=False,
            )
            estimator.fit(
                first.iloc[inner_fit],
                target[fitting][inner_fit],
                cat_features=solution.categorical_columns(first),
            )
            out_of_fold[inner_validation] = estimator.predict(
                first.iloc[inner_validation]
            )
        residual = np.abs(target[fitting] - out_of_fold)
        threshold = np.quantile(residual, 0.95)
        weights = np.where(residual > threshold, 0.25, 1.0)
        prediction, _, _ = fit_pair(first, target[fitting], second, weights)
        record("OOF_top5pct_weight_0.25", prediction, perf_counter() - start)
        report["oof_details"] = dict(
            OOF_threshold=float(threshold),
            downweighted_rows=int((weights < 1).sum()),
            tiny_downweighted=int(
                ((weights < 1) & raw.revenue.iloc[fitting].le(1000).to_numpy()).sum()
            ),
        )
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["budget_masking"]), flush=True)


if __name__ == "__main__":
    main()
