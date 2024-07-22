from typing import Tuple

import numpy as np



def pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Precision-Recall Curve)"""
    sorted_indices = np.argsort(y_prob, kind='mergesort')[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    true_positives = np.cumsum(y_true_sorted)
    total_positives = np.sum(y_true_sorted)


    recalls = true_positives / total_positives
    precisions = true_positives / np.arange(1, len(y_true_sorted) + 1)

    # Find the threshold that satisfies the minimum precision
    valid_recall_indices = np.where(precisions >= min_precision)[0]
    if len(valid_recall_indices) > 0:
        selected_index = np.argmax(recalls[valid_recall_indices])
        selected_threshold = y_prob_sorted[valid_recall_indices[selected_index]]
        selected_recall = recalls[valid_recall_indices[selected_index]]
        return selected_threshold, selected_recall

def sr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Specificity-Recall Curve)"""
    pass


def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Precision-Recall curve and it's (LCB, UCB)"""
    pass


def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Specificity-Recall curve and it's (LCB, UCB)"""
    pass


