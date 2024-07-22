from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import math
from scipy.spatial.distance import cdist


@dataclass
class ImageKMeans:
    n_clusters: int = 5
    init: str | np.ndarray = "random"
    max_iter: int = 100
    random_state: int = 42

    def _image_as_array(self, image: np.ndarray) -> np.ndarray:
        """Convert image to pixel array"""
        return image.reshape(-1, image.shape[-1])

    def _init_centroids(self, X: np.ndarray) -> None:
        """Select N random samples as initial centroids"""
        if isinstance(self.init, str):
            if self.init != 'random':
                raise ValueError(f"Unrecognized str init: {self.init}")
            np.random.seed(self.random_state)
            ind = np.random.choice(len(X), size=self.n_clusters, replace=False)
            self.centroids_ = X[ind]

        if isinstance(self.init, np.ndarray):
            if self.init.shape != (self.n_clusters, 3):
                raise ValueError
            if np.unique(self.init, axis=0).shape[0] != self.n_clusters:
                raise ValueError
            if np.any(self.init < 0) or np.any(self.init > 255):
                raise ValueError

            self.centroids_ = self.init
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise TypeError

    def _assign_centroids(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to the closest centroid"""
        y = np.zeros(len(X))

        for i in range(len(X)):
            #distances = np.linalg.norm(X[i] - self.centroids_, axis=1)
            distances = cdist(X[i].reshape(1, -1), self.centroids_)
            y[i] = np.argmin(distances)

        return y

