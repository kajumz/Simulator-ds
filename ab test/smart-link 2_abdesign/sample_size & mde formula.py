import numpy as np
from scipy import stats


def calculate_sample_size(
        reward_avg: float, reward_std: float, mde: float, alpha: float, beta: float
) -> int:
    """Calculate sample size.

    Parameters
    ----------
    reward_avg: float :
        average reward
    reward_std: float :
        standard deviation of reward
    mde: float :
        minimum detectable effect
    alpha: float :
        significance level
    beta: float :
        type 2 error probability

    Returns
    -------
    int :
        sample size

    """
    assert mde > 0, "mde should be greater than 0"

    # Implement your solution here
    Z_alpha_2 = stats.norm.ppf(1 - alpha / 2)
    Z_beta = stats.norm.ppf(1 - beta)

    # Calculate sample size
    sample_size = (2 * (Z_alpha_2 + Z_beta) ** 2 * (reward_std ** 2)) / (mde*reward_avg) ** 2

    return int(np.ceil(sample_size))



def calculate_mde(
        reward_std: float, sample_size: int, alpha: float, beta: float
) -> float:
    """Calculate minimal detectable effect.

    Parameters
    ----------
    reward_avg: float :
        average reward
    reward_std: float :
        standard deviation of reward
    sample_size: int :
        sample size
    alpha: float :
        significance level
    beta: float :
        type 2 error probability

    Returns
    -------
    float :
        minimal detectable effect

    """

    # Implement your solution here
    Z_alpha_2 = stats.norm.ppf(1 - alpha / 2)
    Z_beta = stats.norm.ppf(1 - beta)

    # Calculate minimal detectable effect (MDE)
    mde = np.sqrt(((Z_alpha_2 + Z_beta) ** 2 * 2 * reward_std ** 2) / sample_size)

    return mde

n_mean = 4200000

