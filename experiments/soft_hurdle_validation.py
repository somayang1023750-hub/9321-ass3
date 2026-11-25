"""Reproduce the saved candidate's two repeats of three-fold training-only CV."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import z5429763 as solution
from experiments.soft_hurdle_model import SoftHurdleRevenueRegressor


def compare_repeat(train, seed):
    truth = np.log1p(train.revenue.to_numpy())
    baseline = np.empty(len(train))
    candidate = np.empty(len(train))
    seen = np.zeros(len(train), dtype=int)
    folds = []
    for fold, (fit, held) in enumerate(StratifiedKFold(
        3, shuffle=True, random_state=seed
    ).split(train, train.rating)):
        model = SoftHurdleRevenueRegressor().fit(train.iloc[fit], truth[fit])
        baseline[held] = np.maximum(0, model.baseline_.predict(train.iloc[held]))
        candidate[held] = model.predict(train.iloc[held])
        seen[held] += 1
        folds.append(dict(
            fold=fold, fitting_rows=len(fit), validation_rows=len(held),
            baseline_RMSLE=float(np.sqrt(np.mean((baseline[held] - truth[held])**2))),
            candidate_RMSLE=float(np.sqrt(np.mean((candidate[held] - truth[held])**2))),
        ))
    assert np.all(seen == 1)
    old_error = (baseline - truth)**2
    new_error = (candidate - truth)**2
    rng = np.random.default_rng(20261115)
    improvements = []
    for _ in range(1000):
        sample = rng.integers(0, len(train), len(train))
        improvements.append(float(
            np.sqrt(old_error[sample].mean()) - np.sqrt(new_error[sample].mean())
        ))
    return dict(
        repeat_seed=seed, folds=folds, pooled_rows=len(train),
        baseline_pooled_RMSLE=float(np.sqrt(old_error.mean())),
        candidate_pooled_RMSLE=float(np.sqrt(new_error.mean())),
        paired_improvement_95CI=np.quantile(improvements, [.025, .975]).tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeat-seeds", type=int, nargs="+", default=[20261115, 20261116])
    args = parser.parse_args()
    solution.THREADS = args.threads
    train = pd.read_csv(args.train_file)
    results = []
    for seed in args.repeat_seeds:
        results.append(compare_repeat(train, seed))
        report = dict(
            train_sha256=hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
            baseline_source_sha256=hashlib.sha256((ROOT / "z5429763.py").read_bytes()).hexdigest(),
            threads=args.threads, results=results,
        )
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
