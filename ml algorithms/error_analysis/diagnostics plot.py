import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def xy_fitted_residuals(y_true, y_pred):
    """Coordinates (x, y) for fitted residuals against true values."""
    residuals = y_true - y_pred
    return y_pred, residuals


def xy_normal_qq(y_true, y_pred):
    """Coordinates (x, y) for normal Q-Q plot."""
    residuals = y_true - y_pred
    sorted_residuals = np.sort(residuals)
    mean = np.mean(sorted_residuals)
    std = np.std(sorted_residuals)
    stan = (sorted_residuals - mean) / std
    n = len(residuals)
    quantiles = np.linspace(0, 1, n, endpoint=False)
    theoretical_quantiles = stats.norm.ppf(quantiles)
    return theoretical_quantiles, stan


def xy_scale_location(y_true, y_pred):
    """Coordinates (x, y) for scale-location plot."""
    residuals = y_true - y_pred
    mean = np.mean(residuals)
    std = np.std(residuals)
    stan = (residuals - mean) / std
    return y_pred, np.sqrt(np.abs(stan))

