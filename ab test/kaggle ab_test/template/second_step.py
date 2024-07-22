"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from tqdm import tqdm


def prepare_dataset(DATA_PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset.
    Load data, split into X and y, one-hot encode categorical

    Parameters
    ----------
    DATA_PATH: str :
        path to the dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        X and y
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["ID"], axis=1)
    y = df.pop("y").values

    # select only numeric columns
    X_num = df.select_dtypes(include="number")

    # select only categorical columns and one-hot encode them
    X_cat = df.select_dtypes(exclude="number")
    X_cat = pd.get_dummies(X_cat)

    # combine numeric and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.fillna(0).values

    return X, y


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Tuple[int, int]],
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 42,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Cross-validation score.

    Parameters
    ----------
    model: Callable :
        model to train (e.g. RandomForestRegressor)

    X: np.ndarray :

    y: np.ndarray :

    cv Union[int, Tuple[int, int]]:
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

    params_list: List[Dict] :
        list of model parameters

    scoring: Callable :
        scoring function (e.g. r2_score)

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    np.ndarray :
        cross-validation scores [n_models x n_folds]

    """
    # YOUR CODE HERE
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, random_state=random_state, shuffle=True)
    else:
        cv = RepeatedKFold(n_splits=cv[0], n_repeats=cv[1], random_state=random_state)
    scores = []

    for params in params_list:
        model_instance = model.set_params(**params)
        cv_scores = []
        #if show_progress:
        #    print(f"Fitting model {i + 1}/{len(params_list)}")

        for train_index, test_index in cv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model_instance.fit(X_train, np.log1p(y_train))
            y_pred = np.expm1(model_instance.predict(X_test))
            score = scoring(y_test, y_pred)
            cv_scores.append(score)
        scores.append(cv_scores)
    return np.array(scores)


def compare_models(
    cv: Union[int, Tuple[int, int]],
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    alpha: float = 0.05,
    show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: Union[int, Tuple[int, int]] :
        (Default value = 5)
        number of folds or (n_folds, n_repeats)
        if int, then KFold is used
        if tuple, then RepeatedKFold is used

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    alpha: float :
        (Default value = 0.05)
        significance level for t-test

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            p_value,
            effect_sign
        }
    """
    res = []
    metrics = cross_val_score(
        model=model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )
    base_score = metrics[0]
    base_score_mean = np.mean(base_score)
    for i, params in enumerate(params_list[1:]):
        score = metrics[i+1]
        avg_score = np.mean(score)
        stat, p_value = ttest_rel(score, base_score)
        effect_sign = 0
        if p_value < alpha and avg_score > base_score_mean:
            effect_sign = 1
        elif p_value < alpha and avg_score < base_score_mean:
            effect_sign = -1
        elif p_value > alpha:
            effect_sign = 0
        result = {
            'model_index': i,
            'avg_score': avg_score,
            'p_value': p_value,
            'effect_sign': effect_sign
        }
        res.append(result)
        #if show_progress:
        #    print(f"Model {i} - Avg Score: {avg_score:.4f} - Effect Sign: {effect_sign}")
    results = sorted(res, key=lambda x: x['avg_score'], reverse=True)
    return results



def run() -> None:
    """Run."""

    data_path = "train.csv"
    random_state = 42
    cv = 5
    params_list = [
        {"max_depth": 10},  # baseline
        {"max_depth": 2},
        {"max_depth": 3},
        {"max_depth": 4},
        {"max_depth": 5},
        {"max_depth": 9},
        {"max_depth": 11},
        {"max_depth": 12},
        {"max_depth": 15},
    ]

    X, y = prepare_dataset(data_path)
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)

    result = compare_models(
        cv=cv,
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        show_progress=True,
    )
    print("KFold")
    print(pd.DataFrame(result))


if __name__ == "__main__":
    run()
