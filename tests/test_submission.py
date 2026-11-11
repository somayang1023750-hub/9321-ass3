"""Contract tests with synthetic data; no assignment test labels influence fitting."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[1] / "z5429763.py"
spec = importlib.util.spec_from_file_location("submission", MODULE)
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)


@pytest.fixture
def movies():
    rng = np.random.default_rng(42)
    count = 120
    df = pd.DataFrame(index=range(count))
    for name in solution.NUMERIC_COLUMNS:
        df[name] = rng.integers(1, 1000, count).astype(float)
    for name in solution.LIST_COLUMNS + solution.CATEGORY_COLUMNS:
        df[name] = ["known a, known b"] * count
    for name in ("title", "original_title", "overview", "tagline"):
        df[name] = ["Some example text"] * count
    df["release_date"] = "2020-04-12"
    df["imdb_id"] = [f"tt{i:07}" for i in range(count)]
    df["vote_average"] = 6.0
    df["status"] = "Released"
    df["revenue"] = df["budget"] ** 2
    df["rating"] = np.resize(["Low", "Medium", "High"], count)
    return df


def test_predictors_do_not_read_targets_or_ids(movies):
    expected = solution.preprocess(movies)
    changed = movies.drop(columns=list(solution.EXCLUDED_COLUMNS))
    pd.testing.assert_frame_equal(expected, solution.preprocess(changed))
    pd.testing.assert_frame_equal(
        solution.regression_preprocess(movies), solution.regression_preprocess(changed)
    )
    changed["rating"] = "POISON"
    changed["revenue"] = -999999
    changed["imdb_id"] = "POISON"
    pd.testing.assert_frame_equal(expected, solution.preprocess(changed))
    pd.testing.assert_frame_equal(
        solution.regression_preprocess(movies), solution.regression_preprocess(changed)
    )


def test_preprocess_handles_missing_unseen_and_extreme_values(movies):
    movies.loc[0, ["budget", "runtime"]] = 0
    movies.loc[0, "release_date"] = "not-a-date"
    movies.loc[1, "release_date"] = None
    movies.loc[0, "cast"] = None
    movies.loc[1, "genres"] = "Previously unseen genre"
    movies.loc[0, "popularity"] = np.inf
    features = solution.preprocess(movies)
    assert features.loc[0, "budget_missing"] == 1
    assert features.loc[0, "runtime_missing"] == 1
    assert features.loc[0, "release_date_missing"] == 1
    assert features.loc[0, "cast_first"] == "__MISSING__"
    assert pd.isna(features.loc[0, "popularity"])
    assert features.filter(like="genre_").loc[1].sum() == 0
    assert not np.isinf(features.select_dtypes(include="number").to_numpy()).any()


def test_predictions_ignore_test_targets_and_support_unseen_categories(
    movies, monkeypatch
):
    monkeypatch.setattr(
        solution, "REGRESSION_PARAMS", dict(iterations=20, depth=3, learning_rate=0.1)
    )
    monkeypatch.setattr(
        solution,
        "CLASSIFICATION_PARAMS",
        dict(iterations=20, depth=3, learning_rate=0.1),
    )
    monkeypatch.setattr(
        solution,
        "REVENUE_CAT_PARAMS",
        dict(iterations=20, depth=3, learning_rate=0.1, max_ctr_complexity=1),
    )
    monkeypatch.setattr(
        solution,
        "REVENUE_LGB_PARAMS",
        dict(n_estimators=20, num_leaves=7, learning_rate=0.1, min_child_samples=5),
    )
    test = movies.iloc[:6].copy()
    test["director"] = "Unseen director"
    revenue, rating = solution.train_predict(movies, test)
    blind_test = test.drop(columns=["rating", "revenue"])
    other_revenue, other_rating = solution.train_predict(movies, blind_test)
    np.testing.assert_array_equal(revenue, other_revenue)
    np.testing.assert_array_equal(rating, other_rating)
    changed_train = movies.copy()
    changed_train["revenue"] = changed_train["revenue"] * 10
    changed_revenue, _ = solution.train_predict(changed_train, blind_test)
    assert not np.allclose(revenue, changed_revenue)
    assert len(revenue) == len(test)
    assert np.isfinite(revenue).all() and (revenue >= 0).all()
    assert set(rating).issubset({"Low", "Medium", "High"})


def test_cli_uses_passed_paths_and_writes_current_directory(movies, tmp_path):
    train_path = tmp_path / "arbitrary training name.csv"
    test_path = tmp_path / "arbitrary prediction name.csv"
    movies.to_csv(train_path, index=False)
    test = movies.iloc[:8].drop(columns=["revenue", "rating"]).copy()
    test["director"] = "A new director"
    test.to_csv(test_path, index=False)
    result = subprocess.run(
        [sys.executable, str(MODULE), str(train_path), str(test_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for task, target in [("regression", "revenue"), ("classification", "rating")]:
        output = pd.read_csv(tmp_path / f"z5429763_{task}.csv")
        assert output.columns.tolist() == ["imdb_id", target]
        assert output.imdb_id.tolist() == test.imdb_id.tolist()
        assert len(output) == len(test)
        assert not output[target].isna().any()
    regression = pd.read_csv(tmp_path / "z5429763_regression.csv")
    classification = pd.read_csv(tmp_path / "z5429763_classification.csv")
    assert np.isfinite(regression.revenue).all() and (regression.revenue >= 0).all()
    assert set(classification.rating).issubset({"Low", "Medium", "High"})


def test_log_inverse_always_finite_and_nonnegative():
    values = solution.revenue_from_log(np.array([-100, 0, 5, 1000], dtype=float))
    assert np.isfinite(values).all()
    assert (values >= 0).all()


def test_notebook_is_executed_and_within_submission_limit():
    import json

    path = MODULE.with_suffix(".ipynb")
    assert path.stat().st_size < 3_000_000
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is not None
        assert cell.get(
            "outputs"
        ), "Expected visible analysis evidence in each code cell"
        assert not any(output["output_type"] == "error" for output in cell["outputs"])


def _load_proposal_experiment():
    path = MODULE.parent / "experiments" / "proposal_validation.py"
    module_spec = importlib.util.spec_from_file_location("proposal_experiment", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_proposal_frequency_does_not_learn_from_validation_batches(movies):
    proposal = _load_proposal_experiment()
    fitting = movies.iloc[:4].copy()
    fitting["director"] = ["A", "A", "B", None]
    validation = movies.iloc[:3].copy()
    validation["director"] = ["A", "UNSEEN", None]
    first = proposal.feature_frame(fitting, ("missing", "release"))
    second = proposal.feature_frame(validation, ("missing", "release"))
    _, encoded, maps = proposal.add_frequency(first, second)
    larger = pd.concat([validation] * 10, ignore_index=True)
    _, encoded_larger, other_maps = proposal.add_frequency(
        first, proposal.feature_frame(larger, ("missing", "release"))
    )
    pd.testing.assert_frame_equal(
        encoded.reset_index(drop=True), encoded_larger.iloc[:3].reset_index(drop=True)
    )
    assert encoded.director_freq.tolist() == [2.0, 0.0, 0.0]
    for name in maps:
        pd.testing.assert_series_equal(maps[name], other_maps[name])
    assert "director_freq" not in first.columns


def test_proposal_features_ignore_targets_and_handle_missing_ratios(movies):
    proposal = _load_proposal_experiment()
    data = movies.copy()
    data.loc[0, ["budget", "runtime"]] = [1.0, 100.0]
    data.loc[1, "budget"] = 0
    data.loc[2, "runtime"] = np.nan
    features = proposal.feature_frame(data, ("missing", "release"))
    blind = data.drop(columns=list(solution.EXCLUDED_COLUMNS))
    pd.testing.assert_frame_equal(
        features, proposal.feature_frame(blind, ("missing", "release"))
    )
    assert features.loc[0, "budget_per_runtime"] < 0
    assert features.loc[0, "budget_per_runtime_missing"] == 0
    assert features.loc[1, "budget_per_runtime"] == -99
    assert features.loc[2, "budget_per_runtime_missing"] == 1
    assert not np.isinf(features.select_dtypes(include="number").to_numpy()).any()


def test_revenue_predictions_are_independent_of_batch_order_and_index(
    movies, monkeypatch
):
    monkeypatch.setattr(
        solution,
        "REVENUE_CAT_PARAMS",
        dict(iterations=20, depth=3, learning_rate=0.1, max_ctr_complexity=1),
    )
    monkeypatch.setattr(
        solution,
        "REVENUE_LGB_PARAMS",
        dict(n_estimators=20, num_leaves=7, learning_rate=0.1, min_child_samples=5),
    )
    model = solution.RevenueRegressor().fit(movies, np.log1p(movies.revenue).to_numpy())
    query = movies.iloc[:4].drop(columns=["revenue", "rating"]).copy()
    query["cast"] = ["new actor", "known a", None, "new actor, known b"]
    query.index = [500, -2, 97, 11]
    expected = model.predict(query)
    np.testing.assert_allclose(
        model.predict(query.iloc[[2, 0, 3, 1]]), expected[[2, 0, 3, 1]], rtol=1e-12
    )
    larger = pd.concat([query] + [query.iloc[[0]]] * 10, ignore_index=True)
    np.testing.assert_allclose(model.predict(larger)[:4], expected, rtol=1e-12)
    for position in range(4):
        np.testing.assert_allclose(
            model.predict(query.iloc[[position]]), expected[[position]], rtol=1e-12
        )
    np.testing.assert_allclose(model.predict(query), expected, rtol=1e-12)
