import numpy as np
from scipy.stats import ttest_ind
from typing import Tuple

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


def aa_test(
        n_simulations: int,
        n_samples: int,
        conversion_rate: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
) -> float:
    """Do the A/A test (simulation)."""

    type_1_errors = 0
    for _ in range(n_simulations):
        cpc_a_no_diff = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        cpc_b_no_diff = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)

        is_sign, _ = t_test(cpc_a_no_diff, cpc_b_no_diff, alpha)
        if is_sign:
            type_1_errors += 1
        # Generate two cpc samples with the same conversion_rate, reward_avg, and reward_std
        # Check t-test and save type 1 error

    # Calculate the type 1 errors rate
    type_1_errors_rate = type_1_errors / n_simulations

    return type_1_errors_rate
