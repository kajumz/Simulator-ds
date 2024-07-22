from typing import Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from sklearn.utils import resample
from scipy.interpolate import interp1d

def pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Precision-Recall Curve)"""
    sorted_indices = np.argsort(y_prob, kind='mergesort')[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    cumulative_positives = np.cumsum(y_true_sorted)
    recall = cumulative_positives / np.sum(y_true_sorted)

    precision = cumulative_positives / np.arange(1, len(y_true_sorted) + 1)

    valid_precision_indices = np.where(precision >= min_precision)[0]

    if len(valid_precision_indices) == 0:
        # If no precision satisfies the condition, return the maximum recall
        return y_prob_sorted[0], recall[0]

    last_valid_precision_index = valid_precision_indices[-1]
    threshold_proba = y_prob_sorted[last_valid_precision_index]
    max_recall = recall[last_valid_precision_index]

    return threshold_proba, max_recall


def sr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Specificity-Recall Curve)"""
    sorted_indices = np.argsort(y_prob, kind='mergesort')[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    num_negatives = np.sum(y_true == 0)
    num_false_positives = 0

    left, right = 0, len(y_prob_sorted) - 1
    best_threshold = y_prob_sorted[0]
    best_recall = 0.0

    while left <= right:
        mid = (left + right) // 2

        if y_true_sorted[mid] == 1:
            right = mid - 1
        else:
            num_false_positives = len(y_prob_sorted[:mid + 1]) - np.sum(y_true_sorted[:mid + 1])
            current_specificity = 1.0 - num_false_positives / num_negatives
            current_recall = np.sum(y_true_sorted[:mid + 1]) / np.sum(y_true_sorted)

            if current_specificity >= min_specificity and current_recall > best_recall:
                best_threshold = y_prob_sorted[mid]
                best_recall = current_recall

            if current_specificity >= min_specificity:
                left = mid + 1
            else:
                right = mid - 1

    return best_threshold, best_recall




def bootstrap_curve(y_true: np.ndarray, y_prob: np.ndarray, curve_function, n_bootstrap: int) -> np.ndarray:
    #np.random.seed(42)
    curves = np.zeros((n_bootstrap, len(y_true) + 1))

    for i in range(n_bootstrap):
        # Bootstrap separately for positive and negative examples
        pos_indices = np.where(y_true == 1)[0]
        neg_indices = np.where(y_true == 0)[0]

        pos_bootstrap_indices = np.random.choice(pos_indices, len(pos_indices), replace=True)
        neg_bootstrap_indices = np.random.choice(neg_indices, len(neg_indices), replace=True)

        bootstrap_indices = np.concatenate([pos_bootstrap_indices, neg_bootstrap_indices])
        bootstrap_curve = curve_function(y_true[bootstrap_indices], y_prob[bootstrap_indices])

        curves[i, :] = bootstrap_curve

    return curves


def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Precision-Recall curve and it's (LCB, UCB)"""
    sorted_indices = np.argsort(y_prob, kind='mergesort')[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    cumulative_positives = np.cumsum(y_true_sorted)
    recall = cumulative_positives / np.sum(y_true_sorted)

    precision = cumulative_positives / np.arange(1, len(y_true_sorted) + 1)

    valid_precision_indices = np.where(precision >= conf)[0]

    if len(valid_precision_indices) == 0:
        # If no precision satisfies the condition, return the maximum recall
        return recall, precision, recall, recall

    last_valid_precision_index = valid_precision_indices[-1]
    threshold_proba = y_prob_sorted[last_valid_precision_index]
    max_recall = recall[last_valid_precision_index]

    precision_values = np.zeros((n_bootstrap, len(precision)))
    recall_values = np.zeros((n_bootstrap, len(recall)))

    # Interpolate the bootstrap curves based on the recall values of the original curve
    for i in range(n_bootstrap):
        pos_bootstrap_indices = np.random.choice(pos_indices, len(pos_indices), replace=True)
        neg_bootstrap_indices = np.random.choice(neg_indices, len(neg_indices), replace=True)
        bootstrap_indices = np.concatenate([pos_bootstrap_indices, neg_bootstrap_indices])

        bootstrap_recall = np.cumsum(y_true[bootstrap_indices]) / np.sum(y_true[bootstrap_indices])
        bootstrap_precision = np.cumsum(y_true[bootstrap_indices]) / np.arange(1, len(bootstrap_indices) + 1)

        interp_function = interp1d(bootstrap_recall, bootstrap_precision, kind='linear', bounds_error=False,
                                   fill_value=(0.0, 1.0))
        interpolated_curve = interp_function(recall)

        precision_values[i, :] = interpolated_curve
        recall_values[i, :] = recall

    precision_lcb = np.percentile(precision_values, (1 - conf) / 2 * 100, axis=0)
    precision_ucb = np.percentile(precision_values, (1 + conf) / 2 * 100, axis=0)

    return recall, precision, precision_lcb, precision_ucb


def specificity_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds, indices = np.unique(y_prob, return_index=True)
    thresholds = np.flip(thresholds)
    indices = np.flip(indices)

    tp = np.cumsum(y_true[indices] == 1)
    fn = np.sum(y_true == 1) - tp
    tn = np.cumsum(y_true[indices] == 0)
    fp = np.sum(y_true == 0) - tn

    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)

    return recall, specificity, thresholds

def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Specificity-Recall curve and it's (LCB, UCB)"""
    recall, specificity, thresholds = specificity_recall_curve(y_true, y_prob)
    curves = bootstrap_curve(y_true, y_prob, specificity_recall_curve, n_bootstrap)

    specificity_values = np.zeros((n_bootstrap, len(specificity)))
    recall_values = np.zeros((n_bootstrap, len(recall)))

    # Interpolate the bootstrap curves based on the recall values of the original curve
    for i, curve in enumerate(curves):
        interp_function = interp1d(recall, curve, kind='linear', bounds_error=False, fill_value=(1.0, 0.0))
        interpolated_curve = interp_function(recall)

        specificity_values[i, :] = interpolated_curve
        recall_values[i, :] = recall

    specificity_lcb = np.percentile(specificity_values, (1 - conf) / 2 * 100, axis=0)
    specificity_ucb = np.percentile(specificity_values, (1 + conf) / 2 * 100, axis=0)

    return recall, specificity, specificity_lcb, specificity_ucb
