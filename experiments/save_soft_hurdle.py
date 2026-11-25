"""Fit and save the soft-partition candidate using only a training CSV.

Usage: python experiments/save_soft_hurdle.py <train_file> --output-dir <directory>
This does not evaluate or open an assignment test file or change the CLI model.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import z5429763 as solution
from experiments.soft_hurdle_model import GATE_PARAMS, SoftHurdleRevenueRegressor


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(train_file, output_dir):
    output = Path(output_dir)
    model_path = output / "soft_hurdle.joblib"
    manifest_path = output / "manifest.json"
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("Saved version exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_file)
    target = np.log1p(train.revenue.to_numpy())
    started = perf_counter()
    model = SoftHurdleRevenueRegressor().fit(train, target)
    fit_seconds = perf_counter() - started
    # Persistence and target-blindness checks, not held-out performance claims.
    sample = train.iloc[::max(1, len(train) // 16)].iloc[:16]
    inputs = sample.drop(columns=["revenue", "rating"], errors="ignore")
    before = model.predict(inputs)
    np.testing.assert_allclose(before, model.predict(sample), rtol=1e-12, atol=1e-12)
    if not np.isfinite(before).all():
        raise ValueError("Saved model produced nonfinite predictions")
    with tempfile.NamedTemporaryFile(
        dir=output, prefix=".saving-", suffix=".joblib", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(model, temporary_path, compress=3)
        restored = joblib.load(temporary_path)
        np.testing.assert_allclose(
            restored.predict(inputs), before, rtol=1e-12, atol=1e-12
        )
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    manifest = dict(
        candidate="soft_hurdle_v1", output="log1p(revenue)",
        training_rows=len(train), low_rows=int(train.revenue.le(1000).sum()),
        normal_rows=int(train.revenue.gt(1000).sum()), income_threshold=1000,
        baseline_fraction=0.5, hurdle_fraction=0.5,
        gate_params=GATE_PARAMS, cat_params=solution.REVENUE_CAT_PARAMS,
        light_params=solution.REVENUE_LGB_PARAMS, training_threads=solution.THREADS,
        train_sha256=sha256(train_file), model_sha256=sha256(model_path),
        source_sha256={
            "z5429763.py": sha256(ROOT / "z5429763.py"),
            "experiments/soft_hurdle_model.py": sha256(ROOT / "experiments/soft_hurdle_model.py"),
        },
        python_version=platform.python_version(),
        package_versions={name: version(name) for name in [
            "numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "joblib"
        ]},
        fit_seconds=fit_seconds, artifact_bytes=model_path.stat().st_size,
        persistence_check="reload predictions agree within 1e-12",
        evaluation="Full-data refit; validation metrics belong to prior OOF fits",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_file")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    save(args.train_file, args.output_dir)


if __name__ == "__main__":
    main()
