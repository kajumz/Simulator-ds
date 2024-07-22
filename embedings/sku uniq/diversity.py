from typing import Tuple
import numpy as np
from kde_uniq import kde_uniqueness

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
    embed = kde_uniqueness(embeddings)
    sc = np.mean(embed)
    gr = True
    if sc < threshold:
        gr = False
    return gr, sc
