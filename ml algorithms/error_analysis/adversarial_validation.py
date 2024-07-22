from typing import Optional

import pandas as pd
import numpy as np
import residuals
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def adversarial_validation(
    classifier: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    quantile: float = 0.1,
    func: Optional[str] = None,
) -> dict:
    X_test = X_test.copy()
    """Adversarial validation residual analysis"""
    #resid = np.empty(shape=len(y_test))
    if func is None:
        resid = y_test - y_pred
        resid = np.abs(resid)
    else:
        func = func or "residuals"
        resid_func = getattr(residuals, func)
        resid = np.abs(resid_func(y_test, y_pred))
        resid = pd.Series(resid, index=y_test.index)
    top_k = np.floor(len(X_test) * quantile).astype(int)

    resid.sort_values(ascending=False, inplace=True)

    top_resid = resid[:top_k]
    idx = top_resid.index

    # Create a Pandas Series with 0 values
    target = pd.Series(0, index=X_test.index)

    # Replace values at top k indices with 1
    target.iloc[idx] = 1

    # Train a classifier to predict if a sample is a worst case or not
    #model = LogisticRegression()
    classifier.fit(X_test, target)

    # Calculate ROC-AUC for adversarial validation classifier
    roc_auc = roc_auc_score(target, classifier.predict_proba(X_test)[:, 1])

    # Calculate feature importances of the adversarial validation classifier
    if hasattr(classifier, "coef_"):
        feature_importances = pd.Series(np.abs(classifier.coef_[0]), index=X_test.columns)
    elif hasattr(classifier, "feature_importances_"):
        feature_importances = pd.Series(classifier.feature_importances_)
    else:
        feature_importances = None


    result = {
        "ROC-AUC": roc_auc,
        "feature_importances": feature_importances,
    }

    return result