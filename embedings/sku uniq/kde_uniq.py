import numpy as np
from sklearn.neighbors import KernelDensity

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
    #kdes = []
    #for dim in range(embeddings.shape[1]):
    #    kde = KernelDensity()
    #    kde.fit(embeddings[:, dim].reshape(-1, 1))
    #    score = np.exp(kde.score_samples(embeddings[:, dim].reshape(-1, 1)))
    #    kdes.append(1/score)
    #return np.array(kdes)
    kde = KernelDensity()  # Вы можете настроить bandwidth

    # Обучение модели KDE на ваших вложениях
    kde.fit(embeddings)

    # Вычисление оценок типичности для каждого элемента
    scores = np.exp(kde.score_samples(embeddings))

    # Вычисление уникальности как обратное значение типичности
    uniqueness = 1 / scores

    return uniqueness

