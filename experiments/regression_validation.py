"""Reproduce the selected regression comparison using only a training CSV.

Usage: python experiments/regression_validation.py <train_file> --output <report.json>
The supplied assignment test file is never opened by this development tool.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import z5429763 as solution


def compare(train, seed):
    fitting, validation = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=seed, stratify=train.rating
    )
    target = np.log1p(train.revenue.to_numpy())
    features = solution.preprocess(train)
    started = perf_counter()
    baseline = solution.make_regressor()
    baseline.fit(
        features.iloc[fitting],
        target[fitting],
        cat_features=solution.categorical_columns(features),
    )
    baseline_prediction = np.maximum(0, baseline.predict(features.iloc[validation]))
    baseline_seconds = perf_counter() - started
    started = perf_counter()
    candidate = solution.RevenueRegressor().fit(train.iloc[fitting], target[fitting])
    prediction = np.maximum(0, candidate.predict(train.iloc[validation]))
    candidate_seconds = perf_counter() - started
    error = (target[validation] - prediction) ** 2
    baseline_error = (target[validation] - baseline_prediction) ** 2
    rng = np.random.default_rng(42)
    bootstrap, improvement = [], []
    for _ in range(1000):
        sample = rng.integers(0, len(validation), len(validation))
        score = float(np.sqrt(error[sample].mean()))
        bootstrap.append(score)
        improvement.append(float(np.sqrt(baseline_error[sample].mean())) - score)
    subgroups = {}
    for name, mask in {
        "budget_unknown": train.budget.iloc[validation].eq(0).to_numpy(),
        "budget_known": train.budget.iloc[validation].gt(0).to_numpy(),
        "revenue_le_1000": train.revenue.iloc[validation].le(1000).to_numpy(),
        "revenue_gt_1000": train.revenue.iloc[validation].gt(1000).to_numpy(),
    }.items():
        subgroups[name] = dict(
            rows=int(mask.sum()),
            RMSLE=float(np.sqrt(error[mask].mean())),
            squared_log_error_share=float(error[mask].sum() / error.sum()),
        )
    return dict(
        seed=seed,
        fitting_rows=len(fitting),
        validation_rows=len(validation),
        baseline_RMSLE=float(np.sqrt(baseline_error.mean())),
        candidate_RMSLE=float(np.sqrt(error.mean())),
        baseline_seconds=baseline_seconds,
        candidate_seconds=candidate_seconds,
        candidate_RMSLE_95_CI=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        improvement_95_CI=np.quantile(improvement, [0.025, 0.975]).tolist(),
        subgroups=subgroups,
        cat_feature_importance=sorted(
            [
                dict(feature=name, importance=float(value))
                for name, value in zip(
                    candidate.cat_model.feature_names_,
                    candidate.cat_model.feature_importances_,
                )
            ],
            key=lambda row: row["importance"],
            reverse=True,
        )[:12],
        lightgbm_feature_gain=sorted(
            [
                dict(feature=name, gain=float(value))
                for name, value in zip(
                    candidate.feature_names,
                    candidate.lgb_model.booster_.feature_importance(
                        importance_type="gain"
                    ),
                )
            ],
            key=lambda row: row["gain"],
            reverse=True,
        )[:12],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train = pd.read_csv(args.train_file)
    results = []
    for seed in [42, 2026]:
        result = compare(train, seed)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
