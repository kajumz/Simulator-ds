from typing import Any
from collections import defaultdict
from sklearn.utils.validation import indexable
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection._split import _BaseKFold


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator < 1] = 1
    return np.mean(numerator / denominator)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(y_pred - y_true) / np.sum(np.abs(y_true))



class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum groups for a single training set.
    test_size : int, default=None
        Number of groups in test
    gap : int, default=0
        Number of groups between train and test sets
    Examples
    --------
    >>> import numpy as np
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                    'b', 'b', 'b', 'b', 'b',\
                    'c', 'c', 'c', 'c',\
                    'd', 'd', 'd',
                    'e', 'e', 'e'])
    >>> splitter = GroupTimeSeriesSplit(n_splits=3, max_train_size=2, gap=1)
    >>> for i, (train_idx, test_idx) in enumerate(
    ...     splitter.split(groups, groups=groups)):
    ...     print(f"Split: {i + 1}")
    ...     print(f"Train idx: {train_idx}, test idx: {test_idx}")
    ...     print(f"Train groups: {groups[train_idx]},
                    test groups: {groups[test_idx]}\n")
    Split: 1
    Train idx: [0 1 2 3 4 5], test idx: [11 12 13 14]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a'], test groups: ['c' 'c' 'c' 'c']

    Split: 2
    Train idx: [ 0  1  2  3  4  5  6  7  8  9 10], test idx: [15 16 17]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b'],
    test groups: ['d' 'd' 'd']

    Split: 3
    Train idx: [ 6  7  8  9 10 11 12 13 14], test idx: [18 19 20]
    Train groups: ['b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c'],
    test groups: ['e' 'e' 'e']
    """

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_samples = len(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap

        # Sort and get unique groups
        group_dict = defaultdict(lambda: [])
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[ind.argsort()]  # Sort unique groups based on their appearance order
        n_groups = len(unique_groups)

        # If test_size is given, use it. Else, derive it from n_groups and n_folds
        group_test_size = self.test_size if self.test_size else n_groups // n_folds

        # Populate group dictionary with indices of each group
        for idx in range(len(groups)):
            group_dict[groups[idx]].append(idx)

        if n_folds > n_groups:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of groups={n_groups}."
            )
        if n_groups - gap - (group_test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={group_test_size} and gap={gap}."
            )

        # Determine where the test set should start for each split
        group_test_starts = range(n_groups - n_splits * group_test_size, n_groups, group_test_size)

        for group_test_start in group_test_starts:
            train_indices = []

            # Determine the end of the training set
            group_train_end = group_test_start - gap

            # If max_train_size is set, use it to limit the training data size
            if self.max_train_size and self.max_train_size < group_train_end:
                train_groups = unique_groups[group_train_end - self.max_train_size:group_train_end]
            else:
                train_groups = unique_groups[:group_train_end]

            # Append indices of each group in the training set
            for train_group in train_groups:
                train_indices.extend(group_dict[train_group])

            # Ensure uniqueness and sort the indices
            train = np.unique(train_indices)
            train.sort()

            test_indices = []
            # Extract the groups that should be part of the test set
            test_groups = unique_groups[group_test_start:group_test_start + group_test_size]

            # Append indices of each group in the test set
            for test_group in test_groups:
                test_indices.extend(group_dict[test_group])

            # Ensure uniqueness and sort the indices
            test = np.unique(test_indices)
            test.sort()

            yield train, test


def best_model() -> Any:
    # YOU CODE IS HERE:
    # type your own sklearn model
    # with custom parameters

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, max_leaf_nodes=2)
    return model