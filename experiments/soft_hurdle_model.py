"""Saved research candidate: 50% baseline plus 50% soft income partition.

Returns log1p(revenue). The official submission entry point is unchanged.
The low/high experts learn only their own training partition; the gate never
reads a prediction frame's target columns.
"""
import numpy as np
from catboost import CatBoostClassifier

import z5429763 as solution

GATE_PARAMS = dict(
    iterations=650, depth=6, learning_rate=0.05, l2_leaf_reg=10,
    max_ctr_complexity=1,
)


class SoftHurdleRevenueRegressor:
    def fit(self, df, log_revenue):
        target = np.asarray(log_revenue, dtype=float)
        self.baseline_ = solution.TargetEncodedRevenueRegressor().fit(df, target)
        low = target <= np.log1p(1000)
        self.active_ = int(low.sum()) >= 2 and int((~low).sum()) >= 2
        if not self.active_:
            return self
        self.feature_builder_ = solution.RevenueRegressor()
        features = self.feature_builder_._features(df, fitting=True)
        self.gate_ = CatBoostClassifier(
            **GATE_PARAMS, loss_function="Logloss", random_seed=solution.SEED,
            thread_count=solution.THREADS, verbose=False, allow_writing_files=False,
        ).fit(
            features, low.astype(int),
            cat_features=solution.categorical_columns(features),
        )
        self.low_ = solution.TargetEncodedRevenueRegressor().fit(
            df.iloc[low], target[low]
        )
        self.high_ = solution.TargetEncodedRevenueRegressor().fit(
            df.iloc[~low], target[~low]
        )
        return self

    def predict(self, df):
        baseline = self.baseline_.predict(df)
        if not self.active_:
            return np.maximum(0, baseline)
        probability = self.gate_.predict_proba(
            self.feature_builder_._features(df)
        )[:, 1]
        low = np.clip(self.low_.predict(df), 0, np.log1p(1000))
        high = np.maximum(np.log1p(1000), self.high_.predict(df))
        mixture = probability * low + (1 - probability) * high
        return np.maximum(0, 0.5 * baseline + 0.5 * mixture)
