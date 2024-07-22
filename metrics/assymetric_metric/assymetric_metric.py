import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """assymetric metric function"""
    size, sum_error = len(y_true), 0
    for i in range(size):
        if y_true[i] < y_pred[i]:
            sum_error += 1 * abs(y_true[i] - y_pred[i])
        else:
            sum_error += 2 * abs(y_true[i] - y_pred[i])
    error = float(sum_error/size)

    return error
