"""Uplift@k metric."""
from sklearn.utils.validation import check_consistent_length

import numpy as np


def uplift_at_k(y_true, uplift, treatment, k=0.3):
    """Compute uplift at first k observations by uplift of the total sample."""
    check_consistent_length(y_true, uplift, treatment)
    y_true, uplift, treatment = (
        np.array(y_true),
        np.array(uplift),
        np.array(treatment),
    )

    order = np.argsort(uplift, kind="mergesort")[::-1]
    k = int(len(y_true) * k) if isinstance(k, float) else k

    score_ctrl = y_true[order][:k][treatment[order][:k] == 0].mean()
    score_trmnt = y_true[order][:k][treatment[order][:k] == 1].mean()

    return score_trmnt - score_ctrl
