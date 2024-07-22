from __future__ import annotations

import numpy as np

def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    y_pred = np.mean(y)
    mse = np.square(np.subtract(y, y_pred)).mean()
    return mse


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    m_l = mse(y_left)
    m_r = mse(y_right)
    return (m_l*len(y_left) + m_r*len(y_right)) / (len(y_left) + len(y_right))


def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node (one feature)"""
    thresholds = np.unique(X[:, feature])
    best_score = float('inf')
    best_threshold = None
    for threshold in thresholds:
        mask = X[:, feature] <= threshold
        y_left = y[mask]
        y_right = y[~mask]
        score = weighted_mse(y_left, y_right)
        if score < best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold



