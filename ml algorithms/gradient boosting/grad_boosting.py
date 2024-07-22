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
            loss='mse',
            verbose=False
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        #if loss == 'mse':
        self.loss = loss
        self.verbose = verbose
        self.trees_ = []
        self.base_pred_ = None


    def _mse(self, y_true, y_pred):
        # YOUR CODE HERE
        loss = np.mean((y_true - y_pred) ** 2)
        grad = (y_pred - y_true)
        return (loss, grad)

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        if isinstance(self.loss, Callable):
            nonlocal loss_func
            loss_func = self.loss
        # 1. make base pred
        y_pred = float(np.mean(y))
        self.base_pred_ = y_pred
        for _ in range(self.n_estimators):
            # 2. compute gradient
            gradient = -(loss_func(y, y_pred)[1])
            # 3. fit a new estimator
            estimator = DecisionTreeRegressor(max_depth=self.max_depth,
                                              min_samples_split=self.min_samples_split)
            estimator.fit(X, gradient)
            # 4. make a prediction
            y_pred += self.learning_rate * estimator.predict(X)
            # 5. save tree
            self.trees_.append(estimator)
        return self


    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        # YOUR CODE HERE
        predictions = np.full(X.shape[0], self.base_pred_)
        # Make predictions using each estimator
        for estimator in self.trees_:
            predictions += self.learning_rate * estimator.predict(X)
        return predictions
