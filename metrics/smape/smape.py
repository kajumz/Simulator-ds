import numpy as np

def smape(y_true: np.array, y_pred: np.array) -> float:
    """"smape realization"""
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred)
    return np.mean(np.nan_to_num(np.divide(2 * num, den)))
