"""Template for user."""
from typing import Tuple
from sklearn.neighbors import KernelDensity
import numpy as np


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    kde = KernelDensity()  # Вы можете настроить bandwidth

    # Обучение модели KDE на ваших вложениях
    kde.fit(embeddings)

    # Вычисление оценок типичности для каждого элемента
    scores = np.exp(kde.score_samples(embeddings))

    # Вычисление уникальности как обратное значение типичности
    uniqueness = 1 / scores

    return uniqueness


def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """

    uniqueness = kde_uniqueness(embeddings)

    # Calculate group diversity as the mean uniqueness
    group_diversity_score = np.mean(uniqueness)

    # Check if the group should be rejected based on the threshold
    reject_group = group_diversity_score < threshold

    return reject_group, group_diversity_score


