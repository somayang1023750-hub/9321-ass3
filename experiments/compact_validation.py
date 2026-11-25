"""Validate the frozen compact submission using only a training CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import z5429763 as solution


def compare(train, seed):
    fit, val = train_test_split(
        np.arange(len(train)), test_size=.2, random_state=seed, stratify=train.rating
    )
    fitting, validation = train.iloc[fit], train.iloc[val]
    started = perf_counter()
    regressor = solution.make_current_regressor().fit(
        fitting, np.log1p(fitting.revenue.to_numpy())
    )
    reg_seconds = perf_counter() - started
    started = perf_counter()
    classifier = solution.make_current_classifier().fit(fitting, fitting.rating)
    cls_seconds = perf_counter() - started
    metrics = {}
    for name, frame in [('train', fitting), ('val', validation)]:
        inputs = frame.drop(columns=['revenue', 'rating'])
        prediction = regressor.predict(inputs)
        labels = classifier.predict(inputs)
        metrics[name] = dict(
            rows=len(frame),
            RMSLE=float(np.sqrt(np.mean((np.log1p(frame.revenue.to_numpy())-prediction)**2))),
            MacroF1=float(f1_score(frame.rating, labels, average='macro')),
            accuracy=float(accuracy_score(frame.rating, labels)),
        )
    return dict(seed=seed,regression_fit_seconds=reg_seconds,
                classification_fit_seconds=cls_seconds,metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,2026,2718])
    args = parser.parse_args()
    train = pd.read_csv(args.train_file)
    report = dict(
        train_sha256=hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(solution.__file__).read_bytes()).hexdigest(),
        selection_seed=42,confirmation_seeds=[2026,2718],
        configuration='Pure compact soft hurdle, fixed before confirmation',
        regression_params=solution.COMPACT_REVENUE_CAT_PARAMS,
        gate_params=solution.COMPACT_GATE_PARAMS,
        classification_params=solution.COMPACT_CLASSIFICATION_PARAMS,
        threads=solution.THREADS,results=[],
    )
    for seed in args.seeds:
        result=compare(train,seed)
        report['results'].append(result)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
