from typing import Tuple, Optional
import numpy as np
from scipy.stats import shapiro
from scipy.stats import ttest_1samp
from scipy.stats import bartlett, levene, fligner


def test_normality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """Normality test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the normality test

    is_rejected : bool
        True if the normality hypothesis is rejected, False otherwise

    """
    res = y_true - y_pred
    _, p_value = shapiro(res)
    is_rejected = p_value < alpha
    return (p_value, is_rejected)


def test_unbiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefer: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Unbiasedness test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    prefer : str, optional (default=None)
        If None or "two-sided", test whether the residuals are unbiased.
        If "positive", test whether the residuals are unbiased or positive.
        If "negative", test whether the residuals are unbiased or negative.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the unbiasedness hypothesis is rejected, False otherwise

    """
    residuals = y_true - y_pred
    if prefer == "positive":
        _, p_value = ttest_1samp(residuals, 0, alternative='greater')
    elif prefer == "negative":
        _, p_value = ttest_1samp(residuals, 0, alternative='less')
    else:
        _, p_value = ttest_1samp(residuals, 0)
    is_rejected = p_value < alpha
    return (p_value, is_rejected)



def test_homoscedasticity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
    test: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Homoscedasticity test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    bins : int, optional (default=10)
        Number of bins to use for the test.
        All bins are equal-width and have the same number of samples, except
        the last bin, which will include the remainder of the samples
        if n_samples is not divisible by bins parameter.

    test : str, optional (default=None)
        If None or "bartlett", perform Bartlett's test for equal variances.
        If "levene", perform Levene's test.
        If "fligner", perform Fligner-Killeen's test.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the homoscedasticity hypothesis is rejected, False otherwise

    """
    residuals = y_pred - y_true
    sorted_indices = np.argsort(y_true)
    sorted_residuals = residuals[sorted_indices]

    bin_indices = np.arange(len(sorted_residuals)) // (len(sorted_residuals) // bins)
    binned_residuals = [sorted_residuals[bin_indices == i] for i in range(bins)]

    if test is None or test == "bartlett":
        _, p_value = bartlett(*binned_residuals)
    elif test == "levene":
        _, p_value = levene(*binned_residuals)
    elif test == "fligner":
        _, p_value = fligner(*binned_residuals)
    else:
        raise ValueError("Invalid test type")

    is_rejected = p_value < alpha
    return p_value, is_rejected
