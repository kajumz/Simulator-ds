from typing import Dict, Tuple, List
import numpy as np


class SimilarItems:
    """sim items"""
    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        items = list(embeddings.keys())
        pair_sims = {}
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sim = np.dot(embeddings[items[i]], embeddings[items[j]]) / (np.linalg.norm(embeddings[items[i]]) * np.linalg.norm(embeddings[items[j]]))
                pair_sims[(items[i], items[j])] = float(round(sim, 8))
        return pair_sims

    @staticmethod
    def knn(
            sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        knn_dict = {}

        for pair, similarity in sim.items():
            item1, item2 = pair
            if item1 not in knn_dict:
                knn_dict[item1] = []
            if item2 not in knn_dict:
                knn_dict[item2] = []
            knn_dict[item1].append((item2, similarity))
            knn_dict[item2].append((item1, similarity))

        for item in knn_dict:
            knn_dict[item] = sorted(knn_dict[item], key=lambda x: x[1], reverse=True)[:top]

        return knn_dict

    @staticmethod
    def knn_price(
            knn_dict: Dict[int, List[Tuple[int, float]]],
            prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        weighted_price_dict = {}

        for item, neighbors in knn_dict.items():
            total_weight = 0
            weighted_sum = 0

            for neighbor, similarity in neighbors:
                weight = (similarity + 1) / 2  # Cosine similarity + 1, normalized to [0, 1]
                total_weight += weight
                weighted_sum += weight * prices[neighbor]

            if total_weight > 0:
                weighted_price_dict[item] = round(weighted_sum / total_weight, 2)
            else:
                weighted_price_dict[item] = prices[item]

        return weighted_price_dict

    def transform(
            embeddings: Dict[int, np.ndarray],
            prices: Dict[int, float],
            top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.
        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        sim = SimilarItems.similarity(embeddings)
        knn_dict = SimilarItems.knn(sim, top)
        knn_price_dict = SimilarItems.knn_price(knn_dict, prices)
        return knn_price_dict
