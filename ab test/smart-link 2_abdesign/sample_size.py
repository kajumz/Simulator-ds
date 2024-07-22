from typing import List, Tuple
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
    return is_sign, float(p_value)


def ab_test(
    n_simulations: int,
    n_samples: int,
    conversion_rate: float,
    mde: float,
    reward_avg: float,
    reward_std: float,
    alpha: float = 0.05,
) -> float:
    """Do the A/B test (simulation)."""

    type_2_errors = 0
    for _ in range(n_simulations):
        cpc_a = cpc_sample(n_samples, conversion_rate, reward_avg, reward_std)
        cpc_b = cpc_sample(n_samples, conversion_rate*(1+mde), reward_avg, reward_std)
        is_sign, _ = t_test(cpc_a, cpc_b, alpha)

        if not is_sign:
            type_2_errors += 1
        # Generate one cpc sample with the given conversion_rate, reward_avg, and reward_std
        # Generate another cpc sample with the given conversion_rate * (1 + mde), reward_avg, and reward_std
        # Check t-test and save type 2 error


    # Calculate the type 2 errors rate
    type_2_errors_rate = type_2_errors / n_simulations

    return type_2_errors_rate


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


def select_sample_size(
        n_samples_grid: List[int],
        n_simulations: int,
        conversion_rate: float,
        mde: float,
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
        beta: float = 0.2,
) -> Tuple[int, float, float]:
    """Select sample size."""
    for n_samples in n_samples_grid:
        # Implement your solution here
        type_1_error = aa_test(n_simulations, n_samples, conversion_rate, reward_avg, reward_std, alpha)
        type_2_error = ab_test(n_simulations, n_samples, conversion_rate, mde, reward_avg, reward_std, alpha)
        if type_1_error <= alpha and type_2_error < beta:
            return n_samples, type_1_error, type_2_error

    raise RuntimeError(
        "Can't find sample size. "
        f"Last sample size: {n_samples}, "
        f"last type 1 error: {type_1_error}, "
        f"last type 2 error: {type_2_error}"
        "Make sure that the grid is big enough."
    )


def select_mde(
        n_samples: int,
        n_simulations: int,
        conversion_rate: float,
        mde_grid: List[float],
        reward_avg: float,
        reward_std: float,
        alpha: float = 0.05,
        beta: float = 0.2,
) -> Tuple[float, float]:
    """Select MDE."""
    for mde in mde_grid:
        # Implement your solution here

        type_2_error = ab_test(n_simulations, n_samples, conversion_rate, mde, reward_avg, reward_std, alpha)
        if type_2_error <= 1 - beta:
            return mde, type_2_error

    raise RuntimeError(
        "Can't find MDE. "
        f"Last MDE: {mde}, "
        f"last type 2 error: {type_2_error}. "
        "Make sure that the grid is big enough."
    )
