from dataclasses import dataclass
import numpy as np

@dataclass
class UpliftTreeNode:
    feature_index: int = None
    threshold: float = None
    value: float = None
    tau: float = None  # uplift estimate for the node
    n_treated: int = None
    n_control: int = None
    left: 'UpliftTreeNode' = None  # left child
    right: 'UpliftTreeNode' = None  # right child

@dataclass
class UpliftTreeRegressor:
    max_depth: int = 3
    min_samples_leaf: int = 1000
    min_samples_leaf_treated: int = 300
    min_samples_leaf_control: int = 300

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> 'UpliftTreeRegressor':
        # Your fitting logic here
        self.tree_ = self._build_tree(X, treatment, y, depth=0)
        return self

    def _build_tree(self, X, treatment, y, depth):
        # Existing code for building the tree

        if depth == self.max_depth or len(X) <= self.min_samples_leaf:
            return UpliftTreeNode(value=self._calculate_uplift_estimate(y, treatment))

        best_feature_index, best_threshold = self._find_best_split(X, treatment, y)

        if best_feature_index is None or best_threshold is None:
            return UpliftTreeNode(value=self._calculate_uplift_estimate(y, treatment))

        data_left, data_right = self._split_data(X, treatment, y, best_feature_index, best_threshold)

        # Check if minimum leaf size is satisfied for treated and control groups separately
        if len(data_left[1]) < self.min_samples_leaf_treated or len(data_left[1]) < self.min_samples_leaf_control \
                or len(data_right[1]) < self.min_samples_leaf_treated or len(data_right[1]) < self.min_samples_leaf_control:
            return UpliftTreeNode(value=self._calculate_uplift_estimate(y, treatment))

        node = UpliftTreeNode(
            feature_index=best_feature_index,
            threshold=best_threshold,
            tau=self._calculate_uplift_estimate(y, treatment),
            n_treated=np.sum(treatment),
            n_control=len(treatment) - np.sum(treatment),
            left=self._build_tree(data_left[0], data_left[1], data_left[2], depth + 1),
            right=self._build_tree(data_right[0], data_right[1], data_right[2], depth + 1)
        )

        return node

    def _calculate_uplift_estimate(self, y, treatment):
        treated_mask = treatment == 1
        control_mask = ~treated_mask

        sum_y_treated = np.sum(y[treated_mask])
        sum_y_control = np.sum(y[control_mask])

        sum_treated = np.sum(treated_mask)
        sum_control = np.sum(control_mask)

        tau = (sum_y_treated / sum_treated) - (sum_y_control / sum_control)

        return tau

    def _split_data(self, X, treatment, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], treatment[left_mask], y[left_mask]), (X[right_mask], treatment[right_mask], y[right_mask])

    def _find_best_split(self, X, treatment, y):
        best_feature_index = None
        best_threshold = None
        best_criterion = -np.inf

        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])

            if len(unique_values) > 10:
                percentiles = np.percentile(unique_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
            else:
                percentiles = np.percentile(unique_values, [10, 50, 90])

            threshold_options = np.unique(percentiles)

            for threshold in threshold_options:
                data_left, data_right = self._split_data(X, treatment, y, feature_index, threshold)

                if len(data_left[1]) < self.min_samples_leaf or len(data_right[1]) < self.min_samples_leaf:
                    continue

                current_criterion = self._criterion(data_left, data_right, y)

                if current_criterion > best_criterion:
                    best_criterion = current_criterion
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _criterion(self, data_left, data_right, y):
        # Calculate DeltaDeltaP
        tau_left = self._calculate_uplift_estimate(data_left[2], data_left[1])
        tau_right = self._calculate_uplift_estimate(data_right[2], data_right[1])

        delta_delta_p = np.abs(tau_left - tau_right)

        return delta_delta_p

# Example usage:
# uplift_regressor = Upl
