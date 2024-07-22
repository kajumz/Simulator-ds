from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind


def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    control_quantile = []
    experiment_quantile = []
    for _ in range(n_bootstraps):
        bootstrapped_sample_control = np.random.choice(control, size=len(control))
        bootstrapped_sample_experiment = np.random.choice(experiment, size=len(experiment))
        q95_control = sorted(bootstrapped_sample_control)[int(quantile*n_bootstraps)]
        q95_experiment = sorted(bootstrapped_sample_experiment)[int(quantile * n_bootstraps)]
        control_quantile.append(q95_control)
        experiment_quantile.append(q95_experiment)
    t_statistic, p_value = ttest_ind(control_quantile, experiment_quantile)
    result = True if p_value < alpha else False
    return p_value, result