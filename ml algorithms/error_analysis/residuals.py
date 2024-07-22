import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residuals"""
    return y_true - y_pred


def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared errors"""
    return (y_true - y_pred) ** 2


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms"""
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)


def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE terms"""
    return 1 - (y_pred / y_true)


def quantile_loss(
    y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.01
) -> np.ndarray:
    """Quantile loss terms"""
    return np.maximum(q * (y_true - y_pred), (1 - q) * (y_pred - y_true))