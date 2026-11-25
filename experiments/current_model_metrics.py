"""Audit the frozen submission on fit rows, a training holdout, and supplied test.

The seed-42 development models use 12803 fitting rows. Separate final models
refit all training rows before the supplied test file is opened. Test targets
are read only for scoring already frozen predictions. No search is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, root_mean_squared_log_error,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import z5429763 as solution


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(frame, revenue, rating):
    return dict(
        rows=len(frame),
        RMSLE=float(root_mean_squared_log_error(frame.revenue, revenue)),
        MacroF1=float(f1_score(frame.rating, rating, average="macro")),
        accuracy=float(accuracy_score(frame.rating, rating)),
        classification=classification_report(
            frame.rating, rating, labels=["Low", "Medium", "High"],
            output_dict=True, zero_division=0,
        ),
    )


def development_metrics(train):
    fit, val = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=42,
        stratify=train.rating,
    )
    fitting, validation = train.iloc[fit], train.iloc[val]
    started = perf_counter()
    regressor = solution.make_current_regressor().fit(
        fitting, np.log1p(fitting.revenue.to_numpy())
    )
    classifier = solution.make_current_classifier().fit(fitting, fitting.rating)
    fit_seconds = perf_counter() - started
    metrics = {}
    for name, frame in [("train", fitting), ("val", validation)]:
        inputs = frame.drop(columns=["revenue", "rating"])
        revenue = solution.revenue_from_log(regressor.predict(inputs))
        rating = classifier.predict(inputs)
        metrics[name] = score(frame, revenue, rating)
        print(name, {key: metrics[name][key] for key in [
            "rows", "RMSLE", "MacroF1", "accuracy"
        ]}, flush=True)
    inputs = validation.drop(columns=["revenue", "rating"])
    truth = np.log1p(validation.revenue.to_numpy())
    new = regressor.predict(inputs)
    groups = {}
    for name, mask in {
        "revenue_le1000": validation.revenue.le(1000).to_numpy(),
        "revenue_gt1000": validation.revenue.gt(1000).to_numpy(),
        "budget_unknown": validation.budget.fillna(0).le(0).to_numpy(),
        "budget_known": validation.budget.gt(0).to_numpy(),
    }.items():
        groups[name] = dict(
            rows=int(mask.sum()),
            current_RMSLE=float(np.sqrt(np.mean((truth[mask] - new[mask]) ** 2))),
        )
    return dict(
        seed=42, stratify="rating", validation_fraction=0.2,
        fitting_rows=len(fit), validation_rows=len(val),
        fit_indices_sha256=hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest(),
        val_indices_sha256=hashlib.sha256(val.astype("<i8").tobytes()).hexdigest(),
        fit_seconds=fit_seconds, metrics=metrics,
        validation_groups=groups,
    )


def final_metrics(train, test_file, output_dir):
    started = perf_counter()
    regressor = solution.make_current_regressor().fit(
        train, np.log1p(train.revenue.to_numpy())
    )
    classifier = solution.make_current_classifier().fit(train, train.rating)
    fit_seconds = perf_counter() - started
    # Open the test file only after the final models have finished fitting.
    test = pd.read_csv(test_file)
    inputs = test.drop(columns=["revenue", "rating"], errors="ignore")
    revenue = solution.revenue_from_log(regressor.predict(inputs))
    rating = classifier.predict(inputs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for task, target, values in [
        ("regression", "revenue", revenue), ("classification", "rating", rating)
    ]:
        path = output / f"z5429763_{task}.csv"
        pd.DataFrame({"imdb_id": test.imdb_id.to_numpy(), target: values}).to_csv(
            path, index=False
        )
        paths[task] = dict(file=path.name, sha256=sha256(path))
    # The predictions and output files above are frozen before test scoring.
    test_metrics = score(test, revenue, rating)
    blind_train = train.drop(columns=["revenue", "rating"])
    train_metrics = score(
        train, solution.revenue_from_log(regressor.predict(blind_train)),
        classifier.predict(blind_train),
    )
    print("test", {key: test_metrics[key] for key in [
        "rows", "RMSLE", "MacroF1", "accuracy"
    ]}, flush=True)
    return dict(
        fitting_rows=len(train), fit_seconds=fit_seconds,
        metrics=dict(full_train=train_metrics, test=test_metrics), outputs=paths,
    )


def audit(train_file, test_file, report_path, output_dir):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path = report_path.with_name(report_path.stem + "_freeze.json")
    frozen = dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        model="CompactHurdleRevenueRegressor + CompactRatingClassifier",
        source_sha256=sha256(solution.__file__), train_sha256=sha256(train_file),
        income_threshold=1000, baseline_fraction=0.0, hurdle_fraction=1.0,
        gate_params=solution.COMPACT_GATE_PARAMS,
        cat_params=solution.COMPACT_REVENUE_CAT_PARAMS,
        light_params=solution.REVENUE_LGB_PARAMS,
        classifier_params=solution.COMPACT_CLASSIFICATION_PARAMS,
        classifier_class_weights="SqrtBalanced", seed=solution.SEED,
        threads=solution.THREADS, python=platform.python_version(),
        packages={name: version(name) for name in [
            "numpy", "pandas", "scikit-learn", "catboost", "lightgbm"
        ]},
    )
    # Freeze all choices before opening the supplied test, including its labels.
    freeze_path.write_text(json.dumps(frozen, indent=2) + "\n")
    train = pd.read_csv(train_file)
    development = development_metrics(train)
    final = final_metrics(train, test_file, output_dir)
    if sha256(solution.__file__) != frozen["source_sha256"]:
        raise RuntimeError("Submission source changed during the audit")
    report = dict(
        frozen=frozen, test_sha256=sha256(test_file), development=development,
        final_refit=final,
        notes=[
            "train/val share the seed-42 development fit; test uses a separate all-train refit.",
            "Train predictions are in-sample, using frozen full-fit maps, not an OOF estimate.",
            "Target-encoding maps used at prediction differ from cross-fitted training features.",
            "The supplied test is scored only; no model selection uses these results.",
            "The tutors' hidden-test metrics and current CSE runtime are unknown.",
        ],
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-dir", required=True)
    args = parser.parse_args()
    audit(args.train_file, args.test_file, args.output, args.prediction_dir)


if __name__ == "__main__":
    main()
