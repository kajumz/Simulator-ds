from dataclasses import dataclass
import numpy as np


@dataclass
class TreeNode:
    left: None
    right: None
    value: float = None
    threshold: float = None
    feature_index: int = None



@dataclass
class UpliftTreeRegressor:
    """

    Parameters
    ----------
    max_depth : np.int :
        maximum depth of tree    
    min_samples_leaf : int :
        minimum count of samples in leaf    
    min_samples_leaf_treated : int :
        minimum count of treated samples in leaf
    min_samples_leaf_control : int :    
        minimum count of control samples in leaf
    Returns
    -------

    """
    max_depth: int = 3
    min_samples_leaf: int = 1000
    min_samples_leaf_treated: int = 300
    min_samples_leaf_control: int = 300


    def split_data_(self, X, treatment, y, feature_index, threshold):
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask
        return (X[left_mask], treatment[left_mask], y[left_mask]), (X[right_mask], treatment[right_mask], y[right_mask])

    def _calculate_uplift_estimate(self, y, treatment):
        treated_mask = treatment == 1
        control_mask = ~treated_mask

        sum_y_treated = np.sum(y[treated_mask])
        sum_y_control = np.sum(y[control_mask])

        sum_treated = np.sum(treated_mask)
        sum_control = np.sum(control_mask)

        tau = (sum_y_treated / sum_treated) - (sum_y_control / sum_control)

        return tau

    def delta_delta_p(self, data_left, data_right):
        tau_l = self._calculate_uplift_estimate(data_left[2], data_left[1])
        tau_r = self._calculate_uplift_estimate(data_right[2], data_right[1])
        res = np.abs(tau_l - tau_r)
        return res




    def build_(self, X, treatment, y, depth):
        if depth == self.max_depth or len(X) <= self.min_samples_leaf:
            return TreeNode(value=self._calculate_uplift_estimate(y, treatment))

        best_criterion = -np.inf
        best_index = None
        best_threshold = None
        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])
            if len(unique_values) > 10:
                percentiles = np.percentile(unique_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
            else:
                percentiles = np.percentile(unique_values, [10, 50, 90])

            threshold_options = np.unique(percentiles)
            for threshold in threshold_options:
                data_left, data_right = self.split_data_(X, treatment, y, feature_index, threshold)
                if len(data_left[1]) < self.min_samples_leaf:
                    continue
                if len(data_right[1]) < self.min_samples_leaf:
                    continue
                if len(data_left[1]) < self.min_samples_leaf_treated:
                    continue
                if len(data_right[1]) < self.min_samples_leaf_treated:
                    continue
                if len(data_left[1]) < self.min_samples_leaf_control:
                    continue
                if len(data_right[1]) < self.min_samples_leaf_control:
                    continue

                crit = self.delta_delta_p(data_left, data_right)
                if crit > best_criterion:
                    best_criterion = crit
                    best_index = feature_index
                    best_threshold = threshold
        if best_index is None or best_threshold is None:
            return TreeNode(value=self._calculate_uplift_estimate(y, treatment))
        node = TreeNode(feature_index=best_index, threshold=best_threshold)

        left_mask = X[:, best_index] <= best_threshold
        right_mask = ~left_mask

        node.left = self.build_(X[left_mask], treatment[left_mask], y[left_mask], depth + 1)
        node.right = self.build_(X[right_mask], treatment[right_mask], y[right_mask], depth + 1)

        return node

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)



    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> UpliftTreeRegressor:
        """Fit model."""
        self.tree_ = self.build_(X, treatment, y, depth=0)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicts for X."""
        return np.array([self._predict_tree(x, self.tree_) for x in X])