import numpy as np
from typing import Optional

class PCA:
    """
    Principal Component Analysis (PCA) is a linear dimensionality reduction technique.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
        If n_components is not set then all components are stored.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

    components_ : ndarray of shape (n_features, n_components)
        Principal axes in feature space, representing the directions of maximum variance.

    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If n_components is not set then all components are stored and the sum of the ratios is equal to 1.0.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

    def fit(self, X: np.ndarray):
        """
        Fit the model with X by performing eigen decomposition on covariance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Calculate mean
        self.mean_ = np.mean(X, axis=0)

        # Centering data
        X = X - self.mean_

        # Calculate covariance matrix
        n = X.shape[0]
        cov_matrix = np.dot(np.transpose(X), X) / (n-1)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors based on eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, idx]

        # Calculate explained variance
        self.explained_variance_ = eigenvalues[idx]
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        # Centering data
        X_centered = X - self.mean_
        comp = self.components_[:, :self.n_components]
        # Project data
        X_new = np.dot(X_centered, comp)

        return X_new

if __name__ == "__main__":
    X = np.array(
        [
            [1.2, 2, 3.2],
            [4.2, 5.3, 6.6],
            [7, 8.1, 9.7],
            [9, 4, 4.5],
        ]
    )

    print("PCA with all components")
    print("--------")
    pca = PCA()
    pca.fit(X)
    X_projected = pca.transform(X)
    print(pca.__dict__)
    print(pca.explained_variance_ratio_.sum())
    print(X.shape, X_projected.shape)
    print(X_projected)
    print("--------\n")

    print("PCA with 1 component")
    print("--------")
    pca = PCA(1)
    pca.fit(X)
    X_projected = pca.transform(X)
    print(pca.__dict__)
    print(pca.explained_variance_ratio_.sum())
    print(X.shape, X_projected.shape)
    print(X_projected)
    print("--------")
