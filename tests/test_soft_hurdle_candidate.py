"""Persistence and target boundaries for the saved research candidate."""
import joblib
import numpy as np
import pandas as pd
import pytest

import z5429763 as solution
from experiments import soft_hurdle_model as candidate


@pytest.fixture
def fitted(monkeypatch):
    monkeypatch.setattr(solution, 'THREADS', 2)
    monkeypatch.setattr(solution, 'REVENUE_CAT_PARAMS', dict(
        iterations=20, depth=3, learning_rate=.1, max_ctr_complexity=1
    ))
    monkeypatch.setattr(solution, 'REVENUE_LGB_PARAMS', dict(
        n_estimators=20, num_leaves=7, min_child_samples=5
    ))
    monkeypatch.setattr(candidate, 'GATE_PARAMS', dict(
        iterations=20, depth=3, learning_rate=.1, max_ctr_complexity=1
    ))
    rng = np.random.default_rng(42)
    size = 60
    frame = pd.DataFrame({name: rng.integers(1, 1000, size).astype(float)
                          for name in solution.NUMERIC_COLUMNS})
    for name in solution.LIST_COLUMNS + solution.CATEGORY_COLUMNS:
        frame[name] = [f'person {i % 3}' for i in range(size)]
    for name in ['title', 'original_title', 'overview', 'tagline']:
        frame[name] = 'Example text'
    frame['release_date'] = '2020-04-12'
    frame['imdb_id'] = [f'tt{i}' for i in range(size)]
    frame['status'] = 'Released'
    target = np.r_[np.linspace(1, 6, 15), np.linspace(8, 16, 45)]
    frame['revenue'] = np.expm1(target)
    frame['rating'] = np.resize(['Low', 'Medium', 'High'], size)
    model = candidate.SoftHurdleRevenueRegressor().fit(frame, target)
    return model, frame, target


def test_saved_candidate_never_reads_prediction_targets(fitted):
    model, frame, _ = fitted
    query = frame.iloc[[2, 20, 7, 45]].copy()
    expected = model.predict(query.drop(columns=['revenue', 'rating']))
    query['revenue'] = -1e30
    query['rating'] = 'POISON'
    np.testing.assert_array_equal(model.predict(query), expected)
    assert np.isfinite(expected).all() and (expected >= 0).all()


def test_saved_candidate_prediction_order_and_batch_invariance(fitted):
    model, frame, _ = fitted
    query = frame.iloc[[4, 25, 1]].copy()
    query.index = [7, 7, -3]
    expected = model.predict(query)
    np.testing.assert_allclose(model.predict(query.iloc[::-1]), expected[::-1])
    extended = pd.concat([query, frame.iloc[:20]], ignore_index=True)
    np.testing.assert_allclose(model.predict(extended)[:3], expected)


def test_saved_candidate_round_trip(fitted, tmp_path):
    model, frame, _ = fitted
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path, compress=3)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.predict(frame.iloc[:8]),
                               model.predict(frame.iloc[:8]), rtol=1e-12, atol=1e-12)


def test_saved_candidate_single_group_fallback(fitted):
    _, frame, target = fitted
    normal = target > np.log1p(1000)
    model = candidate.SoftHurdleRevenueRegressor().fit(frame.iloc[normal], target[normal])
    assert not model.active_
    np.testing.assert_array_equal(model.predict(frame.iloc[:5]),
                                  np.maximum(0, model.baseline_.predict(frame.iloc[:5])))
