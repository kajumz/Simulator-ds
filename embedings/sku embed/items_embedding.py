import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def items_embeddings(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
    """Build items embedding using factorization model.
    The order of items should be the same in the output matrix.

    Args:
        ui_matrix (pd.DataFrame): User-Item matrix of size (N, M)
        dim (int): Dimention of embedding vectors

    Returns:
        np.ndarray: Items embeddings matrix of size (M, dim)
    """
    # Perform Singular Value Decomposition (SVD)
    _, _, Vt = svds(ui_matrix, k=dim)

    # Reverse the order of singular values and corresponding columns of Vt
    #S = np.flip(S)
    #Vt = np.flip(Vt, axis=0)

    # Multiply U and S to obtain the item embeddings
    #tems_vec = U.dot(np.diag(S))

    return Vt.T
