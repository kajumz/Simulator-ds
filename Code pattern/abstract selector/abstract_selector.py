from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PearsonSelector:
    threshold: float = 0.5

    def fit(self, X, y) -> PearsonSelector:
        # Correlation between features and target
        corr = pd.concat([X, y], axis=1).corr(method="pearson")
        corr_target = corr.iloc[:-1, -1]

        self.original_features = X.columns.tolist()
        self.high_corr_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()

        return self

    def transform(self, X):
        return X[self.high_corr_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        return len(self.high_corr_features)

    @property
    def original_features_(self):
        return self.original_features

    @property
    def selected_features_(self):
        return self.high_corr_features


@dataclass
class SpearmanSelector:
    threshold: float = 0.5

    def fit(self, X, y) -> SpearmanSelector:
        corr = pd.concat([X, y], axis=1).corr(method="spearman")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.high_corr_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()

        return self

    def transform(self, X):
        return X[self.high_corr_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        return len(self.high_corr_features)

    @property
    def original_features_(self):
        return self.original_features

    @property
    def selected_features_(self):
        return self.high_corr_features


@dataclass
class VarianceSelector:
    min_var: float = 0.4

    def fit(self, X, y=None) -> VarianceSelector:
        variances = np.var(X, axis=0)
        self.original_features = X.columns.tolist()
        self.high_var_features = X.columns[variances > self.min_var].tolist()
        return self

    def transform(self, X):
        return X[self.high_var_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        return len(self.high_var_features)

    @property
    def original_features_(self):
        return self.original_features

    @property
    def selected_features_(self):
        return self.high_var_features
