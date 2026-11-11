"""Semantic tests for target-free, movie-level entity frequency features.

These tests intentionally use its public fit/transform interface and do not inspect
Counter implementation details or fit any predictive model.
"""

import numpy as np
import pandas as pd
import pytest

import z5429763 as solution

ENTITY_COLUMNS = (
    "cast",
    "production_companies",
    "director",
    "writers",
    "producers",
    "keywords",
)
STATISTICS = ("max", "mean", "known")
EXPECTED_COLUMNS = {
    f"{column}_all_freq_{stat}" for column in ENTITY_COLUMNS for stat in STATISTICS
}


@pytest.fixture
def entity_movies():
    # Deliberately non-contiguous, unsorted indices; repeated tokens within one
    # film must count as only one appearance after strip/casefold normalization.
    return pd.DataFrame(
        {
            "cast": [" Alice , alice, BOB ", "ALICE", None],
            "production_companies": ["StudioA,STUDIOA", "StudioB", "StudioA"],
            "director": ["Alice", None, ""],
            "writers": ["Writer", "WRITER", "   "],
            "producers": ["Producer P", "producer p", np.nan],
            "keywords": ["space|SPACE|road", "space", None],
            "revenue": [10.0, 100_000.0, 9_000_000.0],
            "rating": ["Low", "Medium", "High"],
            "imdb_id": ["tt10", "tt20", "tt30"],
        },
        index=[40, 10, 90],
    )


def test_entity_frequency_loo_and_default_prediction_semantics(entity_movies):
    encoder = solution.EntityFrequencyFeatures()
    assert encoder.fit(entity_movies) is encoder
    full = encoder.transform(entity_movies)
    loo = encoder.transform(entity_movies, leave_one_out=True)

    assert len(full.columns) == 18
    assert set(full.columns) == EXPECTED_COLUMNS
    pd.testing.assert_index_equal(full.index, entity_movies.index)
    pd.testing.assert_index_equal(loo.index, entity_movies.index)

    # Cast counts in complete fitting data: alice=2, bob=1. Within the first
    # fitting film, excluding itself gives alice=1, bob=0.
    np.testing.assert_allclose(
        full.loc[
            40, ["cast_all_freq_max", "cast_all_freq_mean", "cast_all_freq_known"]
        ],
        [np.log1p(2), (np.log1p(2) + np.log1p(1)) / 2, 1.0],
    )
    np.testing.assert_allclose(
        loo.loc[40, ["cast_all_freq_max", "cast_all_freq_mean", "cast_all_freq_known"]],
        [np.log1p(1), np.log1p(1) / 2, 0.5],
    )
    np.testing.assert_allclose(
        loo.loc[10, ["cast_all_freq_max", "cast_all_freq_mean", "cast_all_freq_known"]],
        [np.log1p(1), np.log1p(1), 1.0],
    )
    # Role vocabularies are separate: Alice directs only one fitting movie.
    assert full.loc[40, "director_all_freq_max"] == pytest.approx(np.log1p(1))
    assert loo.loc[40, "director_all_freq_max"] == 0.0
    # Keywords use pipes, not commas; the same deduplication contract applies.
    np.testing.assert_allclose(
        loo.loc[
            40,
            [
                "keywords_all_freq_max",
                "keywords_all_freq_mean",
                "keywords_all_freq_known",
            ],
        ],
        [np.log1p(1), np.log1p(1) / 2, 0.5],
    )


def test_entity_frequency_empty_and_unseen_entities_are_zero(entity_movies):
    encoder = solution.EntityFrequencyFeatures().fit(entity_movies)
    query = pd.DataFrame(
        {column: [None, "   ", "UNSEEN"] for column in ENTITY_COLUMNS},
        index=[902, -3, 77],
    )
    output = encoder.transform(query)
    assert len(output.columns) == 18
    assert set(output.columns) == EXPECTED_COLUMNS
    pd.testing.assert_index_equal(output.index, query.index)
    np.testing.assert_array_equal(output.to_numpy(), np.zeros((3, 18)))
    assert np.isfinite(output.to_numpy()).all()
    # A leave-one-out request must never turn an unseen count into -1.
    np.testing.assert_array_equal(
        encoder.transform(query, leave_one_out=True).to_numpy(), np.zeros((3, 18))
    )


def test_entity_frequency_batch_order_and_index_invariance(entity_movies):
    encoder = solution.EntityFrequencyFeatures().fit(entity_movies)
    query = entity_movies.iloc[[1, 0, 2]].copy()
    query.index = [901, -2, 65]
    expected = encoder.transform(query)

    # An unrelated prediction batch must not update fitting counts.
    unrelated = query.iloc[[0]].copy()
    unrelated["cast"] = "Alice,Alice,Alice"
    larger = pd.concat([query] + [unrelated] * 20, ignore_index=True)
    actual = encoder.transform(larger)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), actual.iloc[:3].reset_index(drop=True)
    )
    # Arbitrary index labels must not introduce pandas alignment errors.
    reset = encoder.transform(query.reset_index(drop=True))
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), reset)
    shuffled_query = query.iloc[[2, 0, 1]]
    pd.testing.assert_frame_equal(
        encoder.transform(shuffled_query), expected.loc[shuffled_query.index]
    )
    duplicate_index_query = query.copy()
    duplicate_index_query.index = [7, 7, 2]
    duplicate_output = encoder.transform(duplicate_index_query)
    pd.testing.assert_index_equal(duplicate_output.index, duplicate_index_query.index)
    pd.testing.assert_frame_equal(
        duplicate_output.reset_index(drop=True), expected.reset_index(drop=True)
    )
    # Repeating transform never changes results, including after other batches.
    pd.testing.assert_frame_equal(encoder.transform(query), expected)


def test_entity_frequency_refit_discards_old_vocabulary(entity_movies):
    encoder = solution.EntityFrequencyFeatures().fit(entity_movies)
    probe = pd.DataFrame({column: ["Alice", "BRAND_NEW"] for column in ENTITY_COLUMNS})
    assert encoder.transform(probe).loc[0, "cast_all_freq_known"] == 1.0

    replacement = pd.DataFrame({column: ["BRAND_NEW"] for column in ENTITY_COLUMNS})
    encoder.fit(replacement)
    actual = encoder.transform(probe)
    fresh = solution.EntityFrequencyFeatures().fit(replacement).transform(probe)
    pd.testing.assert_frame_equal(actual, fresh)
    np.testing.assert_array_equal(actual.iloc[0].to_numpy(), np.zeros(18))
    assert actual.loc[1, "cast_all_freq_max"] == pytest.approx(np.log1p(1))


def test_entity_frequency_ignores_targets_and_ids_and_preserves_input(entity_movies):
    untouched = entity_movies.copy(deep=True)
    blind = entity_movies[list(ENTITY_COLUMNS)].copy()
    poisoned = entity_movies.copy(deep=True)
    poisoned["revenue"] = [np.nan, -1e20, 1e20]
    poisoned["rating"] = ["POISON", None, "POISON"]
    poisoned["imdb_id"] = ["UNRELATED"] * len(poisoned)

    normal_encoder = solution.EntityFrequencyFeatures().fit(entity_movies)
    blind_encoder = solution.EntityFrequencyFeatures().fit(blind)
    poisoned_encoder = solution.EntityFrequencyFeatures().fit(poisoned)
    expected = normal_encoder.transform(entity_movies)
    pd.testing.assert_frame_equal(expected, blind_encoder.transform(blind))
    pd.testing.assert_frame_equal(expected, poisoned_encoder.transform(poisoned))
    pd.testing.assert_frame_equal(expected, normal_encoder.transform(poisoned))
    assert not set(["revenue", "rating", "imdb_id"]).intersection(expected.columns)
    pd.testing.assert_frame_equal(entity_movies, untouched)


def test_entity_frequency_unicode_casefold_and_within_film_deduplication(entity_movies):
    fitting = entity_movies.iloc[:2].copy()
    fitting["cast"] = [" Straße , STRASSE ", "strasse"]
    fitting["keywords"] = ["Straße| STRASSE ", "strasse"]
    encoder = solution.EntityFrequencyFeatures().fit(fitting)
    query = fitting.iloc[:1].copy()
    query["cast"] = "sTrAsSe"
    query["keywords"] = "straße"
    output = encoder.transform(query)
    loo = encoder.transform(fitting, leave_one_out=True)
    for column in ["cast", "keywords"]:
        assert output.iloc[0][f"{column}_all_freq_max"] == pytest.approx(np.log1p(2))
        assert output.iloc[0][f"{column}_all_freq_mean"] == pytest.approx(np.log1p(2))
        assert output.iloc[0][f"{column}_all_freq_known"] == 1.0
        assert loo.iloc[0][f"{column}_all_freq_max"] == pytest.approx(np.log1p(1))
        assert loo.iloc[0][f"{column}_all_freq_known"] == 1.0
