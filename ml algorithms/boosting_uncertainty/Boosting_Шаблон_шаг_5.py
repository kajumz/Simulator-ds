"""Solution for boosting uncertainty problem"""

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class PredictionDict:
    pred: np.ndarray = np.array([])
    uncertainty: np.ndarray = np.array([])
    pred_virt: np.ndarray = np.array([])
    lcb: np.ndarray = np.array([])
    ucb: np.ndarray = np.array([])


def virtual_ensemble_iterations(
    model: GradientBoostingRegressor, k: int = 20
) -> List[int]:
    n_est = model.n_estimators_
    iter = []
    for i in range(n_est // 2 - 1, n_est, k):
        iter.append(i)
    return iter


def virtual_ensemble_predict(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> np.ndarray:
    num_objects = X.shape[0]
    iterations = virtual_ensemble_iterations(model, k)
    num_models = len(iterations)
    predictions = np.zeros((num_objects, num_models))

    staged_predictions = model.staged_predict(X)
    current_pred_index = 0

    for i, idx in enumerate(iterations):
        for _ in range(idx - current_pred_index):
            next(staged_predictions)  # Skip unnecessary predictions
            current_pred_index += 1
        predictions[:, i] = next(staged_predictions)
        current_pred_index += 1

    return predictions


def predict_with_uncertainty(
    model: GradientBoostingRegressor, X: np.ndarray, k: int = 20
) -> PredictionDict:
    # Get predictions and uncertainties from the original model
    #pred = model.predict(X)

    virtual_pred = virtual_ensemble_predict(model, X, k)
    pred_virt = np.mean(virtual_pred, axis=1)

    # Calculate uncertainty as the variance of predictions from the virtual ensemble
    uncertainty = np.var(virtual_pred, axis=1)

    # Calculate lower and upper confidence bounds
    lcb = pred_virt - 3 * np.sqrt(uncertainty)
    ucb = pred_virt + 3 * np.sqrt(uncertainty)

    prediction_dict = PredictionDict(
        pred=virtual_pred,
        uncertainty=uncertainty,
        pred_virt=pred_virt,
        lcb=lcb,
        ucb=ucb
    )

    return prediction_dict
