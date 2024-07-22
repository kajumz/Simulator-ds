import numpy as np
from scipy.stats import ttest_ind


def cpc_sample(
    n_samples: int, conversion_rate: float, reward_avg: float, reward_std: float
) -> np.ndarray:
    """Sample data."""
    # model binomial dist
    cvr_samples = np.random.binomial(1, conversion_rate, n_samples)

    # model normal dist
    cpa_samples = np.random.normal(reward_avg, reward_std, n_samples)

    cpc = cvr_samples * cpa_samples

    return cpc


def t_test(cpc_a: np.ndarray, cpc_b: np.ndarray, alpha=0.05
) -> Tuple[bool, float]:
    """Perform t-test.

    Parameters
    ----------
    cpc_a: np.ndarray :
        first samples
    cpc_b: np.ndarray :
        second samples
    alpha :
         (Default value = 0.05)

    Returns
    -------
    Tuple[bool, float] :
        True if difference is significant, False otherwise
        p-value
    """
    t_stat, p_value = ttest_ind(cpc_a, cpc_b)
    is_sign = bool(p_value < alpha)
    return is_sign, p_value
