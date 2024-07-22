from typing import List, Tuple
from scipy import stats

def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    mean_control = sum(control) / len(control)
    mean_experiment = sum(experiment) / len(experiment)

    variance_control = sum((x-mean_control)**2 for x in control) / (len(control) - 1)
    variance_experiment = sum((x - mean_experiment) ** 2 for x in experiment) / (len(experiment) - 1)

    standart_error = ((variance_control/len(control)) + (variance_experiment/len(experiment)))**0.5

    t_statistic = (mean_control - mean_experiment) / standart_error
    degrees_of_freedom = len(control) + len(experiment) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=degrees_of_freedom))
    #t_statistic, p_value = stats.ttest_ind(control, experiment)
    result = True if p_value < alpha else False
    return p_value, result
