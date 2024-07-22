"""Solution's template for user."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 
    num_neighbors: int :
        number of neighbors to estimate uniqueness    

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    neighbors_model = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean')
    neighbors_model.fit(embeddings)

    # Находим расстояния и индексы ближайших соседей для каждого товара
    distances, indices = neighbors_model.kneighbors(embeddings)

    # Вычисляем уникальность как среднее расстояние до ближайших соседей
    uniqueness = np.mean(distances, axis=1)

    return uniqueness
