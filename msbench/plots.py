"""Publication-style figures written to the results directory."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402


def save_roc_plot(predictions: Dict[str, Dict[str, np.ndarray]], outdir: Path) -> None:
    plt.figure(figsize=(9, 7))
    for name, pred in predictions.items():
        y_true, y_proba = pred["y_true"], pred["y_proba"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Holdout ROC Curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "holdout_roc_curves.png", dpi=300)
    plt.close()


def save_leaderboard_plot(results: pd.DataFrame, outdir: Path) -> None:
    plot_df = results.sort_values("MS_Research_Score", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["Model"], plot_df["MS_Research_Score"])
    plt.xlabel("MS Research Score / 100")
    plt.title("Model Leaderboard — Exploratory Composite Score")
    for i, score in enumerate(plot_df["MS_Research_Score"]):
        plt.text(score + 0.3, i, f"{score:.2f}", va="center")
    plt.tight_layout()
    plt.savefig(outdir / "ms_research_score_leaderboard.png", dpi=300)
    plt.close()


def save_metric_heatmap(results: pd.DataFrame, outdir: Path) -> None:
    metrics = ["AUC_ROC", "PR_AUC", "Sensitivity", "Specificity", "F1", "Brier"]
    hm = results.set_index("Model")[metrics].copy()
    hm["Brier"] = 1 - hm["Brier"]  # shown as 1 - Brier so higher is better
    plt.figure(figsize=(10, max(5, 0.45 * len(hm))))
    im = plt.imshow(hm.values, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Score, except Brier shown as 1 - Brier")
    plt.xticks(np.arange(len(metrics)), ["AUC", "PR-AUC", "Sens", "Spec", "F1", "1-Brier"], rotation=30, ha="right")
    plt.yticks(np.arange(len(hm.index)), hm.index)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            plt.text(j, i, f"{hm.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Holdout Performance Heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "holdout_metrics_heatmap.png", dpi=300)
    plt.close()
