import torch
import torch.nn.functional as F

def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 5.0,
) -> torch.Tensor:
    """
    Computes the triplet loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        anchor (torch.Tensor): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (torch.Tensor): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (torch.Tensor): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The triplet loss
    """
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    loss = torch.mean(torch.clamp(distance_positive - distance_negative + margin, min=0))
    return loss
