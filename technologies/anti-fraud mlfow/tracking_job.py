import os
from typing import Any
from typing import Tuple

import fire
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

#IDENTIFIER = f'antifraud-{os.environ.get("a-kajumov", "default")}'
IDENTIFIER = 'antifraud-a-kajumov'
TRACKING_URI = "https://mlflow.lab.karpov.courses"
mlflow.set_tracking_uri(uri=TRACKING_URI)


def recall_at_precision(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_precision: float = 0.95,
) -> float:
    """Compute recall at precision

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_precision (float, optional): Min precision for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    metric = recall[precision >= min_precision].max()
    return metric


def recall_at_specificity(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_specificity: float = 0.95,
) -> float:
    """Compute recall at specificity

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_specificity (float, optional): Min specificity for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    metric = tpr[1-fpr >= min_specificity].max()
    return metric


def curves(true_labels: np.ndarray, pred_scores: np.ndarray) -> Tuple[np.ndarray]:
    """Return ROC and FPR curves

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores

    Returns:
        Tuple[np.ndarray]: ROC and FPR curves
    """

    def fig2numpy(fig: Any) -> np.ndarray:
        fig.canvas.draw()
        img = fig.canvas.buffer_rgba()
        img = np.asarray(img)
        return img

    pr_curve = PrecisionRecallDisplay.from_predictions(true_labels, pred_scores)
    pr_curve = fig2numpy(pr_curve.figure_)

    roc_curve = RocCurveDisplay.from_predictions(true_labels, pred_scores)
    roc_curve = fig2numpy(roc_curve.figure_)

    return pr_curve, roc_curve


def job(
    train_path: str = "",
    test_path: str = "",
    target: str = "target",
):
    """Model training job

    Args:
        train_path (str): Train dataset path
        test_path (str): Test dataset path
        target (str): Target column name
    """
    mlflow.set_experiment(experiment_name=IDENTIFIER)
    mlflow.start_run()
    mlflow.set_tags(tags={"task_type" : "anti-fraud",
                          "framework" : "sklearn"})

    FEATURES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11',
                'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21',
                'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30']
    TARGET = 'target'
    mlflow.log_param('features', FEATURES)
    mlflow.log_param('target', TARGET)

    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    model = IsolationForest(n_estimators=150)
    model.fit(train_dataset.drop('target', axis=1))

    mlflow.log_param('model_type', model.__class__.__name__)

    test_targets = test_dataset['target']
    pred_scores = -model.score_samples(test_dataset.drop('target', axis=1))

    roc_auc = roc_auc_score(test_targets, pred_scores)
    recall_at_precision_095 = recall_at_precision(test_targets, pred_scores)
    recall_at_specificity_095 = recall_at_specificity(test_targets, pred_scores)

    mlflow.log_metric('roc_auc', roc_auc)
    mlflow.log_metric('recall_precision_95', recall_at_precision_095)
    mlflow.log_metric('recall_specificity_95', recall_at_specificity_095)

    mlflow.log_artifact(local_path=train_path, artifact_path="data/")
    mlflow.log_artifact(local_path=test_path, artifact_path="data/")

    pr_curve, roc_curve = curves(test_targets, pred_scores)
    mlflow.log_image(pr_curve, artifact_file="metrics/pr.png")
    mlflow.log_image(roc_curve, artifact_file="metrics/roc.png")

    mlflow.sklearn.log_model(model,
                             artifact_path=IDENTIFIER,
                             registered_model_name=IDENTIFIER)

    mlflow.end_run()


if __name__ == "__main__":
    fire.Fire(job)
