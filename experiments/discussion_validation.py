"""Compare frozen pre-discussion and selected models using only a training CSV.

No assignment test file is opened. Selection used split 42; configurations were
then frozen for 2026 and the additional split 2718. Splits overlap and are not
independent external test sets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import z5429763 as solution


CLASSES = np.array(["High", "Low", "Medium"])


def macro_f1(truth, prediction):
    counts = np.bincount(3 * truth + prediction, minlength=9).reshape(3, 3)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    return float(np.divide(
        2 * counts.diagonal(), denominator,
        out=np.zeros(3), where=denominator > 0,
    ).mean())


def compare(train, seed, bootstraps=1000):
    fitting, validation = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=seed, stratify=train.rating
    )
    first, second = train.iloc[fitting], train.iloc[validation]
    log_target = np.log1p(train.revenue.to_numpy())
    started = perf_counter()
    old_regressor = solution.RevenueRegressor().fit(first, log_target[fitting])
    old_regression = np.maximum(0, old_regressor.predict(second))
    old_reg_seconds = perf_counter() - started
    started = perf_counter()
    new_regressor = solution.TargetEncodedRevenueRegressor().fit(
        first, log_target[fitting]
    )
    new_regression = np.maximum(0, new_regressor.predict(second))
    new_reg_seconds = perf_counter() - started

    started = perf_counter()
    features = solution.preprocess(first)
    old_classifier = solution.make_classifier().fit(
        features, first.rating, cat_features=solution.categorical_columns(features)
    )
    old_classification = old_classifier.predict(solution.preprocess(second)).reshape(-1)
    old_cls_seconds = perf_counter() - started
    started = perf_counter()
    new_classifier = solution.EnrichedRatingClassifier().fit(first, first.rating)
    new_classification = new_classifier.predict(second)
    new_cls_seconds = perf_counter() - started

    old_error = (log_target[validation] - old_regression) ** 2
    new_error = (log_target[validation] - new_regression) ** 2
    truth = second.rating.to_numpy()
    labels = np.searchsorted(CLASSES, truth)
    old_labels = np.searchsorted(CLASSES, old_classification)
    new_labels = np.searchsorted(CLASSES, new_classification)
    rng = np.random.default_rng(20261115)
    reg_gain, cls_gain = [], []
    for _ in range(bootstraps):
        sample = rng.integers(0, len(validation), len(validation))
        reg_gain.append(float(np.sqrt(old_error[sample].mean()) - np.sqrt(new_error[sample].mean())))
        cls_gain.append(macro_f1(labels[sample], new_labels[sample]) - macro_f1(labels[sample], old_labels[sample]))
    groups = {}
    for name, mask in {
        "budget_unknown": second.budget.fillna(0).le(0).to_numpy(),
        "budget_known": second.budget.gt(0).to_numpy(),
        "revenue_le1000": second.revenue.le(1000).to_numpy(),
        "revenue_gt1000": second.revenue.gt(1000).to_numpy(),
    }.items():
        groups[name] = dict(
            rows=int(mask.sum()),
            old_RMSLE=float(np.sqrt(old_error[mask].mean())),
            new_RMSLE=float(np.sqrt(new_error[mask].mean())),
        )
    importance = sorted(
        [dict(feature=name, importance=float(value)) for name, value in zip(
            new_regressor.cat_model.feature_names_,
            new_regressor.cat_model.feature_importances_,
        )], key=lambda row: row["importance"], reverse=True,
    )[:15]
    return dict(
        seed=seed, fitting_rows=len(fitting), validation_rows=len(validation),
        old_RMSLE=float(np.sqrt(old_error.mean())),
        new_RMSLE=float(np.sqrt(new_error.mean())),
        old_MacroF1=float(f1_score(truth, old_classification, average="macro")),
        new_MacroF1=float(f1_score(truth, new_classification, average="macro")),
        reg_improvement_95CI=np.quantile(reg_gain, [0.025, 0.975]).tolist(),
        cls_improvement_95CI=np.quantile(cls_gain, [0.025, 0.975]).tolist(),
        old_reg_seconds=old_reg_seconds, new_reg_seconds=new_reg_seconds,
        old_cls_seconds=old_cls_seconds, new_cls_seconds=new_cls_seconds,
        classification_labels=CLASSES.tolist(),
        old_confusion_matrix=confusion_matrix(truth, old_classification, labels=CLASSES).tolist(),
        new_confusion_matrix=confusion_matrix(truth, new_classification, labels=CLASSES).tolist(),
        regression_groups=groups, regression_importance=importance,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 2718])
    args = parser.parse_args()
    train = pd.read_csv(args.train_file)
    results = []
    for seed in args.seeds:
        row = compare(train, seed)
        results.append(row)
        report = dict(
            train_sha256=hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
            results=results,
        )
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
