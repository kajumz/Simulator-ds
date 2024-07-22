from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y_true, y_pred, n_bootstraps):
    aucs = []
    for _ in range(n_bootstraps - 5000):
        # Генерируем бутстреп-выборку
        while True:
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) == 2:
                break  # Прерываем цикл, если в выборке есть оба класса
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]
        auc_bootstrap = roc_auc_score(y_true_bootstrap, y_pred_bootstrap)
        aucs.append(auc_bootstrap)
    return aucs

def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""
    np.random.seed(42)
    y_pred = classifier.predict_proba(X)[:, 1]
    auc_original = roc_auc_score(y, y_pred)

    # Получить оценки ROC-AUC на бутстреп-выборках
    bootstrap_aucs = bootstrap_auc(y, y_pred, n_bootstraps)

    # Рассчитать доверительный интервал
    alpha = 1 - conf
    lower_bound = np.percentile(bootstrap_aucs, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))

    return (lower_bound, upper_bound)
