from typing import Tuple

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> Tuple[int, int, int, int]:
    """Calculate confusion matrix."""
    # YOUR CODE HERE
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for true, pred in zip(y_true, y_pred):
        if pred >= threshold:
            if true == 1:
                TP += 1
            else:
                FP += 1
        else:
            if true == 0:
                TN += 1
            else:
                FN += 1

    return TP, TN, FP, FN


def specificity(TN: int, FP: int) -> float:
    """Calculate specificity."""
    # YOUR CODE HERE
    return TN / (TN+FP)


def test():
    """Test function."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7])
    threshold = 0.5
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred, threshold)

    assert TP == 5
    assert TN == 4
    assert FP == 1
    assert FN == 0

    assert np.allclose(specificity(TN, FP), 0.8)
    print("All tests passed.")


if __name__ == "__main__":
    test()