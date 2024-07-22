import numpy as np


def triplet_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the triplet loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        anchor (np.ndarray): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (np.ndarray): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (np.ndarray): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The triplet loss
    """
    distance_positive = np.sqrt(np.sum(np.square(anchor - positive), axis=1))
    distance_negative = np.sqrt(np.sum(np.square(anchor - negative), axis=1))

    # Compute the triplet loss
    loss = np.mean(np.maximum(distance_positive - distance_negative + margin, 0))
    return loss
