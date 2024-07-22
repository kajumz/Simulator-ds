from typing import List
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
def evaluate(model, embeddings: List[List[float]], labels: List[int], cv=5) -> List[float]:
    """Compute Cross-Entropy Loss for each fold"""
    kf = KFold(n_splits=cv)
    scoring = make_scorer(log_loss, needs_proba=True)
    scores = cross_val_score(model, embeddings, labels, cv=kf, scoring=scoring)
    return scores

