from typing import List

import numpy as np


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

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
    relevance_sort = sorted(relevance, reverse=True)
    if method == 'standard':
        score_dcg = 0
        ideal_score = 0
        for i in range(1, k + 1):
            score_dcg += (relevance[i - 1]) / np.log2(i + 1)
            ideal_score += (relevance_sort[i - 1]) / np.log2(i + 1)
        score = float(score_dcg / ideal_score)
        return score
    elif method == 'industry':
        score_dcg = 0
        ideal_score = 0
        for i in range(1, k + 1):
            score_dcg += ((2 ** relevance[i - 1]) - 1) / np.log2(i + 1)
            ideal_score += ((2 ** relevance_sort[i - 1]) - 1) / np.log2(i + 1)
        score = float(score_dcg / ideal_score)
        return score
    else:
        raise ValueError
