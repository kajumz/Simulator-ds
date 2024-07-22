import numpy as np

def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""
    y_pred = np.mean(y)
    mse = np.square(np.subtract(y, y_pred)).mean()
    return mse


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    m_l = mse(y_left)
    m_r = mse(y_right)
    return (m_l*len(y_left) + m_r*len(y_right)) / (len(y_left) + len(y_right))

X_t = X[:, feature]
    tr = (X_t, y)


    best_threshold = 100000000
    l = [np.mean(X_t), np.min(X_t), np.max(X_t)]
    for threshold in l:
        y_l = [i for i in X if i < threshold]
        X_r = [i for i in X if i >= threshold]
        mse_w = weighted_mse(X_l, X_r)
        best_threshold = min(best_threshold, mse_w)