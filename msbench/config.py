"""Benchmark configuration and the MS Research Score weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    random_seed: int = 42
    test_size: float = 0.20
    outer_cv_folds: int = 5
    inner_cv_folds: int = 3
    n_bootstrap: int = 2000
    top_variance: int = 1000
    k_best_features: int = 200
    smote_k_neighbors: int = 3
    n_jobs: int = -1
    data_file: str = "data/GSE17048_series_matrix.txt.gz"
    results_dir: str = "results"
    positive_label_name: str = "MS"
    negative_label_name: str = "HC"
    external_models: Optional[str] = None


# Weights for the composite MS Research Score. These are exploratory: treat them
# as a reporting convenience, not a clinically validated instrument. They sum to 1.
WEIGHTS = {
    "AUC_ROC": 0.25,
    "PR_AUC": 0.15,
    "Sensitivity": 0.25,
    "Specificity": 0.15,
    "F1": 0.10,
    "Calibration": 0.10,
}
