import numpy as np
from typing import List
from sklearn.metrics import log_loss

def fairness(residuals: np.ndarray) -> float:
    """Compute Gini fairness of array of values"""
    ab = np.abs(residuals)
    #sorted_residuals = np.sort(ab)
    n = len(residuals)
    su = 0
    for i in range(n):
        for j in range(i+1, n):
            su += np.abs(ab[i] - ab[j])


    gini = su / (n ** 2 * np.mean(ab))
    return 1 - gini


def best_prediction(
    y_true: np.ndarray, y_preds: List[np.ndarray], fairness_drop: float = 0.05
) -> int:
    """Find index of best model"""
    log_losses = []
    fairness_scores = []

    for y_pred in y_preds:
        residuals = y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        fairness_score = fairness(residuals)
        fairness_scores.append(fairness_score)

        lo = log_loss(y_true, y_pred)
        log_losses.append(lo)

    max_fairness_drop = fairness_scores[0] * (1 - fairness_drop)
    best_model_index = 0

    for i in range(1, len(fairness_scores)):
        if fairness_scores[i] > max_fairness_drop and log_losses[i] < log_losses[best_model_index]:
            best_model_index = i

    return best_model_index