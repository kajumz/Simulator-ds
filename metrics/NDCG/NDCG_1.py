from typing import List

import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    score = 0
    if method == 'standard':
        for i in range(1, k+1):
            score += (relevance[i-1]) / np.log2(i+1)
        return score
    elif method == 'industry':
        for i in range(1, k+1):
            score += ((2**relevance[i-1]) - 1) / np.log2(i+1)
        return score
    else:
        raise ValueError
