from typing import Optional

import numpy as np
import pandas as pd
import residuals


def best_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k best cases according to the given function"""
    if mask is not None:
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]
    y_test_num = y_test.to_numpy()
    y_pred_num = y_pred.to_numpy()
    resid = np.empty(shape=len(y_test_num))
    if func == 'residuals':
        resid = residuals.residuals(y_test_num, y_pred_num)
    elif func == 'squared_errors':
        resid = residuals.squared_errors(y_test_num, y_pred_num)
    elif func == 'logloss':
        resid = residuals.logloss(y_test_num, y_pred_num)
    elif func == 'ape':
        resid = residuals.ape(y_test_num, y_pred_num)
    elif func is None:
        resid = y_test_num - y_pred_num
    resid_a = np.abs(resid)
    resid_series = pd.Series(resid, index=y_test.index)
    sorted_indices = np.argsort(resid_a)
    top_k_indices = sorted_indices[:top_k]
    result = {
        "X_test": X_test.iloc[top_k_indices],
        "y_test": y_test.iloc[top_k_indices],
        "y_pred": y_pred.iloc[top_k_indices],
        "resid": resid_series.iloc[top_k_indices],
    }
    return result


def worst_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k worst cases according to the given function"""
    if mask is not None:
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]

    y_test_num = y_test.to_numpy()
    y_pred_num = y_pred.to_numpy()
    resid = np.empty(shape=len(y_test_num))
    if func == 'residuals':
        resid = residuals.residuals(y_test_num, y_pred_num)
    elif func == 'squared_errors':
        resid = residuals.squared_errors(y_test_num, y_pred_num)
    elif func == 'logloss':
        resid = residuals.logloss(y_test_num, y_pred_num)
    elif func == 'ape':
        resid = residuals.ape(y_test_num, y_pred_num)
    elif func is None:
        resid = y_test_num - y_pred_num
    resid_a = np.abs(resid)
    resid_series = pd.Series(resid_a, index=y_test.index)
    sorted_indices = np.argsort(resid_a)[::-1]
    top_k_indices = sorted_indices[:top_k]
    result = {
        "X_test": X_test.iloc[top_k_indices],
        "y_test": y_test.iloc[top_k_indices],
        "y_pred": y_pred.iloc[top_k_indices],
        "resid": resid_series.iloc[top_k_indices],
    }
    return result
