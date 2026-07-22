"""Evaluation metrics, the composite MS Research Score, and bootstrap CIs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import WEIGHTS


def safe_predict_proba(estimator: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return estimator.predict(X).astype(float)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else np.nan


def compute_ms_research_score(metrics: Dict[str, float]) -> float:
    """Exploratory composite score out of 100.

    The calibration component is 1 - Brier, so higher is better.
    """
    score = (
        metrics["AUC_ROC"] * 100 * WEIGHTS["AUC_ROC"]
        + metrics["PR_AUC"] * 100 * WEIGHTS["PR_AUC"]
        + metrics["Sensitivity"] * 100 * WEIGHTS["Sensitivity"]
        + metrics["Specificity"] * 100 * WEIGHTS["Specificity"]
        + metrics["F1"] * 100 * WEIGHTS["F1"]
        + max(0.0, (1.0 - metrics["Brier"])) * 100 * WEIGHTS["Calibration"]
    )
    return round(float(score), 2)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC_ROC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan,
        "PR_AUC": average_precision_score(y_true, y_proba),
        "Brier": brier_score_loss(y_true, y_proba),
    }
    metrics["MS_Research_Score"] = compute_ms_research_score(metrics)
    return metrics


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    random_seed: int,
    ci: float = 0.95,
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    scores: List[float] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_func(y_true[idx], y_pred[idx], y_proba[idx]))
        except Exception:
            continue
    if not scores:
        return np.nan, np.nan
    alpha = (1 - ci) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))
