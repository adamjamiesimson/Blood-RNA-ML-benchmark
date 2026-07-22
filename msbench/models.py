"""Model registry, the leakage-safe pipeline, and external-model loading."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .config import Config

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Select the top-k highest-variance features inside CV folds.

    Fitting this inside the pipeline keeps feature selection from seeing the
    held-out fold, which whole-dataset selection would leak. It is unsupervised
    by design.
    """

    def __init__(self, k: int = 1000):
        self.k = k

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = np.asarray(X, dtype=float)
        variances = np.nanvar(X, axis=0)
        k = min(self.k, X.shape[1])
        self.selected_idx_ = np.argsort(variances)[-k:]
        self.selected_idx_.sort()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, self.selected_idx_]

    def get_support(self) -> np.ndarray:
        return self.selected_idx_


def make_preprocessing_pipeline(model: Any, cfg: Config) -> ImbPipeline:
    """Leakage-safe pipeline: every fitted step sees only the training fold.

    Imputation, variance selection, scaling, supervised selection and SMOTE are
    all fitted per fold; SMOTE runs on training data only via the imblearn
    pipeline.
    """
    return ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("topvar", TopVarianceSelector(k=cfg.top_variance)),
            ("scaler", StandardScaler()),
            ("kbest", SelectKBest(score_func=f_classif, k=cfg.k_best_features)),
            ("smote", SMOTE(random_state=cfg.random_seed, k_neighbors=cfg.smote_k_neighbors)),
            ("model", model),
        ]
    )


def model_grid_registry(cfg: Config) -> Dict[str, Tuple[ImbPipeline, Dict[str, List[Any]]]]:
    registry: Dict[str, Tuple[ImbPipeline, Dict[str, List[Any]]]] = {
        "Logistic Regression": (
            make_preprocessing_pipeline(
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=cfg.random_seed), cfg
            ),
            {
                "kbest__k": [50, 100, 200],
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        ),
        "SVM": (
            make_preprocessing_pipeline(
                SVC(probability=True, class_weight="balanced", random_state=cfg.random_seed), cfg
            ),
            {
                "kbest__k": [50, 100, 200],
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale"],
            },
        ),
        "Random Forest": (
            make_preprocessing_pipeline(
                RandomForestClassifier(class_weight="balanced", random_state=cfg.random_seed), cfg
            ),
            {
                "kbest__k": [100, 200],
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "Gradient Boosting": (
            make_preprocessing_pipeline(GradientBoostingClassifier(random_state=cfg.random_seed), cfg),
            {
                "kbest__k": [100, 200],
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "KNN": (
            make_preprocessing_pipeline(KNeighborsClassifier(), cfg),
            {
                "kbest__k": [50, 100, 200],
                "model__n_neighbors": [3, 5, 7, 11],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Naive Bayes": (
            make_preprocessing_pipeline(GaussianNB(), cfg),
            {
                "kbest__k": [50, 100, 200],
                "model__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7],
            },
        ),
        "Neural Network": (
            make_preprocessing_pipeline(MLPClassifier(max_iter=2000, random_state=cfg.random_seed), cfg),
            {
                "kbest__k": [50, 100, 200],
                "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "model__alpha": [1e-4, 1e-3, 1e-2],
                "model__learning_rate": ["constant", "adaptive"],
            },
        ),
    }

    if HAS_XGBOOST:
        registry["XGBoost"] = (
            make_preprocessing_pipeline(
                XGBClassifier(
                    random_state=cfg.random_seed,
                    eval_metric="logloss",
                    objective="binary:logistic",
                    n_jobs=1,
                ),
                cfg,
            ),
            {
                "kbest__k": [100, 200],
                "model__n_estimators": [100, 300],
                "model__max_depth": [2, 3, 5],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        )
    else:
        print("XGBoost not installed. Skipping XGBoost model (pip install xgboost, plus libomp on macOS).")

    return registry


def load_external_models(
    path: Optional[str], cfg: Config
) -> Tuple[Dict[str, Tuple[Any, Dict[str, List[Any]]]], Dict[str, Any]]:
    """Load external models from a Python module or a directory of joblib files.

    Returns (to_tune, pretrained):
      - to_tune:    name -> (estimator, param_grid) — tuned and evaluated like a built-in
      - pretrained: name -> fitted estimator — evaluated as-is
    """
    to_tune: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {}
    pretrained: Dict[str, Any] = {}
    if not path:
        return to_tune, pretrained

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"External models path not found: {p}")

    # Python module defining EXTERNAL_MODELS = {name: estimator or (estimator, param_grid)}
    if p.is_file() and p.suffix == ".py":
        spec = importlib.util.spec_from_file_location("external_models_module", str(p))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        models = getattr(module, "EXTERNAL_MODELS", None)
        if models is None or not isinstance(models, dict):
            raise ValueError("Python external models file must define an EXTERNAL_MODELS dict")
        for name, val in models.items():
            if isinstance(val, (tuple, list)) and len(val) == 2:
                est, grid = val
            else:
                est, grid = val, {}
            to_tune[name] = (est, grid or {})
        return to_tune, pretrained

    # Directory: load any joblib/pkl files as pretrained models
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in {".joblib", ".pkl"}:
                try:
                    pretrained[f.stem] = joblib.load(str(f))
                except Exception as e:
                    print(f"Failed to load pretrained model {f}: {e}")
        return to_tune, pretrained

    # Single pickled model
    if p.is_file() and p.suffix.lower() in {".joblib", ".pkl"}:
        pretrained[p.stem] = joblib.load(str(p))
        return to_tune, pretrained

    raise ValueError(
        "Unsupported external models path. Use a .py module, a directory of "
        ".joblib/.pkl files, or a single .pkl/.joblib file."
    )
