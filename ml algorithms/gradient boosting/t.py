import numpy as np
import pandas as pd
from typing import Callable
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            loss="mse",
            verbose=False
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.trees_ = []
        self.base_pred_ = None

    def _mse(self, y_true, y_pred):
        loss = np.mean((y_true - y_pred) ** 2)
        grad = (y_pred - y_true)
        return (loss, grad)

    def _mae(self, y_true, y_pred):
        loss = np.mean(np.abs(y_true - y_pred))
        grad = np.sign(y_pred - y_true)
        return (loss, grad)

    def _huber(self, y_true, y_pred, delta=1.0):
        diff = y_true - y_pred
        mask = np.abs(diff) <= delta
        squared_loss = 0.5 * diff ** 2
        linear_loss = delta * (np.abs(diff) - 0.5 * delta)
        loss = np.mean(np.where(mask, squared_loss, linear_loss))
        grad = np.where(mask, diff, delta * np.sign(diff))
        return (loss, grad)

    def fit(self, X, y):
        y_pred = float(np.mean(y))
        self.base_pred_ = y_pred

        if isinstance(self.loss, Callable):
            loss_func = self.loss
        elif self.loss == "mse":
            loss_func = self._mse
        elif self.loss == "mae":
            loss_func = self._mae
        elif self.loss == "huber":
            loss_func = self._huber
        else:
            raise ValueError("Unsupported loss function: {}".format(self.loss))

        for _ in range(self.n_estimators):
            gradient = -(loss_func(y, y_pred)[1])
            estimator = DecisionTreeRegressor(max_depth=self.max_depth,
                                              min_samples_split=self.min_samples_split)
            estimator.fit(X, gradient)
            y_pred += self.learning_rate * estimator.predict(X)
            self.trees_.append(estimator)
        return self

    def predict(self, X):
        predictions = np.full(X.shape[0], self.base_pred_)
        for estimator in self.trees_:
            predictions += self.learning_rate * estimator.predict(X)
        return predictions
