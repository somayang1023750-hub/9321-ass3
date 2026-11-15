"""Cross-fitting leakage boundaries and frozen prediction maps."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
import z5429763 as solution


@pytest.fixture
def credits():
    return pd.DataFrame({
        name: ['Alice,alice,BOB', 'alice', '', None, 'BOB', 'Unknown']
        for name in solution.CrossFittedEntityEncoder.columns
    },index=[7,7,-1,90,12,3])


def test_whole_held_fold_targets_do_not_enter_its_encodings(credits):
    target=np.arange(len(credits),dtype=float)
    expected=solution.CrossFittedEntityEncoder(folds=3).fit_transform(credits,target)
    for _,held in KFold(n_splits=3,shuffle=True,random_state=solution.SEED).split(credits):
        poisoned=target.copy();poisoned[held]+=1_000_000
        actual=solution.CrossFittedEntityEncoder(folds=3).fit_transform(credits,poisoned)
        np.testing.assert_allclose(actual.iloc[held],expected.iloc[held])
    pd.testing.assert_index_equal(expected.index,credits.index)


def test_smoothed_means_deduplicate_films_and_keep_roles_separate(credits):
    x=credits.copy();x.loc[:,'director']='Other'
    encoder=solution.CrossFittedEntityEncoder(smoothing=2).fit(x,np.arange(6.))
    query=x.iloc[:1].copy();query.loc[:,'cast']='ALICE';query.loc[:,'director']='ALICE'
    output=encoder.transform(query)
    # Alice appears in two films, not three duplicate credits: prior=2.5.
    assert output.cast_target_mean_0.iloc[0]==pytest.approx((0+1+2*2.5)/4)
    assert output.director_target_mean_0.iloc[0]==pytest.approx(2.5)


def test_prediction_is_target_blind_frozen_and_batch_invariant(credits):
    encoder=solution.CrossFittedEntityEncoder().fit(credits,np.arange(6.))
    query=credits.iloc[[4,1]].copy();expected=encoder.transform(query)
    query['revenue']=[-1e30,1e30];query['rating']=['POISON','POISON'];query['imdb_id']='POISON'
    pd.testing.assert_frame_equal(expected,encoder.transform(query))
    expanded=pd.concat([query,credits]*3,ignore_index=True)
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),encoder.transform(expanded).iloc[:2].reset_index(drop=True))
    pd.testing.assert_frame_equal(expected.iloc[::-1],encoder.transform(query.iloc[::-1]))
    for col in solution.CrossFittedEntityEncoder.columns: query[col]='UNSEEN'
    np.testing.assert_allclose(encoder.transform(query),2.5)


def test_refit_clears_old_targets_and_maps(credits):
    encoder=solution.CrossFittedEntityEncoder().fit(credits,np.arange(6.))
    other=credits.copy()
    for col in solution.CrossFittedEntityEncoder.columns:other[col]='new_entity'
    encoder.fit(other,np.repeat(99.,6))
    np.testing.assert_allclose(encoder.transform(credits),99.)


def test_probability_encoding_uses_only_supplied_task_targets(credits):
    target=np.eye(3)[[0,1,2,0,1,2]]
    values=solution.CrossFittedEntityEncoder(folds=3).fit_transform(credits,target)
    assert np.isfinite(values.to_numpy()).all()
    assert ((values.to_numpy()>=0)&(values.to_numpy()<=1)).all()
    for role in solution.CrossFittedEntityEncoder.columns:
        np.testing.assert_allclose(values[[f'{role}_target_mean_{i}' for i in range(3)]].sum(axis=1),1.)


def test_single_row_rejected_instead_of_self_encoding(credits):
    with pytest.raises(ValueError,match='at least two'):
        solution.CrossFittedEntityEncoder().fit_transform(credits.iloc[:1],[1.])
