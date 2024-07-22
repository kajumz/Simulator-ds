from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None

@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        y_pred = np.mean(y)
        mse = np.square(np.subtract(y, y_pred)).mean()
        return mse

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        m_l = self._mse(y_left)
        m_r = self._mse(y_right)
        return (m_l * len(y_left) + m_r * len(y_right)) / (len(y_left) + len(y_right))


    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_score = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                y_left = y[mask]
                y_right = y[~mask]
                score = self._weighted_mse(y_left, y_right)

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return (best_feature, best_threshold)

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        n_samples = len(X)
        mse = self._mse(y)
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            value = round(np.mean(y))
            return Node(mse=mse, value=value, n_samples=n_samples)

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            value = round(np.mean(y))
            return Node(value=value)

        mask = X[:, best_feature] <= best_threshold
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        left_child = self._split_node(X_left, y_left, depth + 1)
        right_child = self._split_node(X_right, y_right, depth + 1)

        #n_samples = len(X)
        #mse = self._mse(y)
        node = Node(
            feature=best_feature,
            threshold=best_threshold,
            n_samples=n_samples,
            value=round(np.mean(y)),
            mse=mse,
            left=left_child,
            right=right_child
        )

        return node
    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        if self.tree_ is None:
            return ""
        else:
            data = self._as_json(self.tree_)
            return data

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node is None:
            return None

        if node.left is None and node.right is None:
            s = f'{{"value": {node.value}, "n_samples": {node.n_samples}, "mse": {round(node.mse, 2) if node.mse is not None else None}}}'
        else:
            s = f'{{"feature": {node.feature}, "threshold": {node.threshold}, "n_samples": {node.n_samples}, "mse": {round(node.mse, 2) if node.mse is not None else None}, "left": {self._as_json(node.left)}, "right": {self._as_json(node.right)}}}'
        return s

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        # YOUR CODE HERE
        y_pred = np.array([self._predict_one_sample(feature) for feature in X])
        return y_pred

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        # YOUR CODE HERE
        node = self.tree_
        while node.left is not None and node.right is not None:
            if features[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
