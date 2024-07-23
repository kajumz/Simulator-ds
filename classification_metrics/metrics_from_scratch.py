import numpy as np
from typing import Tuple


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
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


def accuracy(TP: int, TN: int, FP: int, FN: int) -> float:
    """Calculate accuracy."""
    # YOUR CODE HERE
    return (TP+TN) / (TP+TN+FP+FN)


def precision(TP: int, FP: int) -> float:
    """Calculate precision."""
    # YOUR CODE HERE
    return TP / (TP+FP)


def recall(TP: int, FN: int) -> float:
    """Calculate recall."""
    # YOUR CODE HERE
    return TP / (TP+FN)


def f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score."""
    # YOUR CODE HERE
    return 2 * (precision*recall) / (precision+recall)


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

    
    assert np.allclose(accuracy(TP, TN, FP, FN), 0.9)

    pr = precision(TP, FP)
    re = recall(TP, FN)
    assert np.allclose(pr, 0.8333333333333334)
    assert np.allclose(re, 1)
    assert np.allclose(f1_score(0.8333333333333334, 1), 0.9090909090909091)
    print("All tests passed.")


if __name__ == "__main__":
    test()
