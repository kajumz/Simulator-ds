from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BaseSelector(ABC):
    """ABC class for class: PearsonSelector, SpearmanSelector, VarianceSelector"""
    @abstractmethod
    def fit(self, X, y):
        pass

    def transform(self, X):
        return X[self.high_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        return len(self.high_features)

    @property
    def original_features_(self):
        return self.original_features

    @property
    def selected_features_(self):
        return self.high_features


@dataclass
class PearsonSelector(BaseSelector):
    """PearsonSelector realization"""
    threshold: float = 0.5

    def fit(self, X, y) -> PearsonSelector:
        corr = pd.concat([X, y], axis=1).corr(method="pearson")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.high_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()
        return self


@dataclass
class SpearmanSelector(BaseSelector):
    """SpearmanSelector realization"""
    threshold: float = 0.5

    def fit(self, X, y) -> SpearmanSelector:
        corr = pd.concat([X, y], axis=1).corr(method="spearman")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.high_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()
        return self


@dataclass
class VarianceSelector(BaseSelector):
    """Variance realization"""
    min_var: float = 0.4

    def fit(self, X, y=None) -> VarianceSelector:
        variances = np.var(X, axis=0)
        self.original_features = X.columns.tolist()
        self.high_features = X.columns[variances > self.min_var].tolist()
        return self
