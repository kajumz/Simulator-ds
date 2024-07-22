"""Solution for Kaggle AB2."""
from typing import Tuple

import numpy as np
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm
from scipy.stats import ttest_rel


class SequentialForwardSelector:
    """
    Sequential forward selection.
    Algorithm selects features one by one, each time adding the feature that
    improves the model the most.

    Parameters
    ----------
    model: estimator :
        ML model, e.g. LinearRegression
    cv: cross-validation generator :
        cross-validation generator, e.g. KFold, RepeatedKFold
    max_features: int :
        maximum number of features to select
    verbose: int :
        (Default value = 0)
        verbosity level

    Attributes
    ----------
    n_features_: int :
        number of features in the dataset
    selected_features_: List[int] :
        list of selected features, ordered by index
    n_selected_features_: int :
        number of selected features
    """

    def __init__(
        self,
        model,
        cv,
        max_features: int = 10,
        alpha: float = 0.05,
        bonferroni: bool = True,
        verbose: int = 0,
    ) -> None:
        """Initialize SequentialForwardSelector."""
        self.model = model
        self.cv = cv
        self.max_features = max_features
        self.verbose = verbose
        self.alpha = alpha
        self.bonferroni = bonferroni
        self.n_features_ = None
        self.selected_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X: np.ndarray :
            features
        y: np.ndarray :
            target
        """
        self.n_features_ = X.shape[1]

        included_feats = []
        excluded_feats = list(range(X.shape[1]))

        scores_a = cross_val_score(
            DummyRegressor(), X, y, scoring="r2", cv=self.cv, n_jobs=-1
        )

        for _ in (
            tqdm(range(self.max_features)) if self.verbose else range(self.max_features)
        ):
            iter_results = []
            scores_b = 0
            for feat in excluded_feats:
                subset = list(included_feats) + [feat]
                scores_b = cross_val_score(
                    self.model, X[:, subset], y, scoring="r2", cv=self.cv, n_jobs=-1
                )
                #if scores_b.mean() < scores_a.mean():
                #    continue

                _, p_value = ttest_rel(scores_b, scores_a, alternative='greater')
                if p_value < (self.alpha / len(excluded_feats)):
                    iter_results.append((feat, scores_b))

            if not iter_results:
                break

            # take best

            best_result = max(iter_results, key=lambda x: sum(x[1]) / len(x))

            # update current solution
            included_feats.append(best_result[0])
            excluded_feats.remove(best_result[0])
            scores_a = best_result[1]

        a = sorted(list(included_feats))
        self.selected_features_ = a

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce the dataset to selected features.

        Parameters
        ----------
        X: np.ndarray :
            features

        Returns
        -------
        X: np.ndarray :
            reduced dataset
        """
        assert self.selected_features_ is not None, "Fit the model first"
        return X[:, self.selected_features_]

    @property
    def n_selected_features_(self):
        """Number of selected features."""
        assert self.selected_features_ is not None, "Fit the model first"
        return len(self.selected_features_)


def generate_dataset(
    n_samples: int = 10_000,
    n_features: int = 50,
    n_informative: int = 10,
    random_state: int = 42,
) -> Tuple:
    """
    Generate dataset.

    Parameters
    ----------
    n_samples: int :
        (Default value = 10_000)
        number of samples
    n_features: int :
        (Default value = 50)
        number of features
    n_informative: int :
        (Default value = 10)
        number of informative features, other features are noise
    random_state: int :
        (Default value = 42)
        random state for reproducibility

    Returns
    -------
    X: np.ndarray :
        features
    y: np.ndarray :
        target

    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=100,
        random_state=random_state,
        n_informative=n_informative,
        bias=100,
        shuffle=True,
    )
    return X, y


def run() -> None:
    """Run."""
    random_state = 42
    n_samples = 10_000
    n_features = 50
    n_informative = 5
    max_features = 10
    n_splits = 3
    n_repeats = 10

    # generate data
    X, y = generate_dataset(n_samples, n_features, n_informative, random_state)

    # define model and cross-validation
    model = LinearRegression()
    cv = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    # baseline
    scores = cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=-1)
    print(f"Baseline features count: {X.shape[1]}")
    print(f"Baseline R2 score: {scores.mean():.4f}")

    selector = SequentialForwardSelector(model, cv, max_features, verbose=1)
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    scores = cross_val_score(model, X_transformed, y, scoring="r2", cv=cv, n_jobs=-1)

    print(f"Features: {selector.selected_features_}")
    print(f"Features count: {selector.n_selected_features_}")
    print(f"Mean R2 score: {scores.mean():.4f}")


if __name__ == "__main__":
    run()