#!/usr/bin/env python3
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Example external model registry.
#   python run_benchmark.py --external-models examples/external_models.py
# Each entry is either an estimator instance or an (estimator, param_grid) pair.
# For pretrained models, save them as .joblib/.pkl and pass the file or directory.

EXTERNAL_MODELS = {
    "External Logistic Regression": (
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "kbest__k": [50, 100, 300],
        },
    ),
    "External Random Forest": (
        RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 90],
        },
    ),
}
