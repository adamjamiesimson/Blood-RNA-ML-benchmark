"""Nested cross-validation, holdout evaluation, and the run orchestrator."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from imblearn.pipeline import Pipeline as ImbPipeline

from .banner import render as render_banner
from .config import Config, WEIGHTS
from .data import get_xy, load_geo_series_matrix
from .metrics import (
    bootstrap_metric_ci,
    evaluate_predictions,
    safe_predict_proba,
    specificity_score,
)
from .models import load_external_models, make_preprocessing_pipeline, model_grid_registry
from .plots import save_leaderboard_plot, save_metric_heatmap, save_roc_plot

# The estimators are noisy under grid search on a small, wide dataset; silence the
# convergence and numeric chatter without hiding real errors.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn version-churn notices
np.seterr(all="ignore")


def _color(text: str, code: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def nested_cv_evaluate_model(
    name: str,
    pipeline: ImbPipeline,
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Nested CV: outer loop estimates performance, inner loop tunes parameters."""
    outer_cv = StratifiedKFold(n_splits=cfg.outer_cv_folds, shuffle=True, random_state=cfg.random_seed)
    inner_cv = StratifiedKFold(n_splits=cfg.inner_cv_folds, shuffle=True, random_state=cfg.random_seed)

    fold_rows: List[Dict[str, Any]] = []
    all_true: List[int] = []
    all_pred: List[int] = []
    all_proba: List[float] = []

    start = time.time()
    for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=cfg.n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        y_proba = safe_predict_proba(best_estimator, X_valid)
        y_pred = (y_proba >= 0.5).astype(int)
        fold_metrics = evaluate_predictions(y_valid, y_pred, y_proba)
        fold_metrics.update(
            {
                "Model": name,
                "Fold": fold,
                "Best_Params": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        fold_rows.append(fold_metrics)
        all_true.extend(y_valid.tolist())
        all_pred.extend(y_pred.tolist())
        all_proba.extend(y_proba.tolist())

    elapsed = time.time() - start
    y_true_arr = np.array(all_true)
    y_pred_arr = np.array(all_pred)
    y_proba_arr = np.array(all_proba)
    pooled = evaluate_predictions(y_true_arr, y_pred_arr, y_proba_arr)

    ci_auc = bootstrap_metric_ci(
        y_true_arr, y_pred_arr, y_proba_arr,
        lambda yt, yp, pr: roc_auc_score(yt, pr),
        cfg.n_bootstrap, cfg.random_seed,
    )
    ci_sens = bootstrap_metric_ci(
        y_true_arr, y_pred_arr, y_proba_arr,
        lambda yt, yp, pr: recall_score(yt, yp, zero_division=0),
        cfg.n_bootstrap, cfg.random_seed,
    )
    ci_spec = bootstrap_metric_ci(
        y_true_arr, y_pred_arr, y_proba_arr,
        lambda yt, yp, pr: specificity_score(yt, yp),
        cfg.n_bootstrap, cfg.random_seed,
    )

    summary = {
        "Model": name,
        **pooled,
        "AUC_95CI": f"[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]",
        "Sensitivity_95CI": f"[{ci_sens[0]:.3f}, {ci_sens[1]:.3f}]",
        "Specificity_95CI": f"[{ci_spec[0]:.3f}, {ci_spec[1]:.3f}]",
        "Runtime_sec": round(elapsed, 2),
        "Outer_CV_Folds": cfg.outer_cv_folds,
        "Inner_CV_Folds": cfg.inner_cv_folds,
    }
    return summary, pd.DataFrame(fold_rows)


def fit_final_model(
    pipeline: ImbPipeline,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> GridSearchCV:
    inner_cv = StratifiedKFold(n_splits=cfg.inner_cv_folds, shuffle=True, random_state=cfg.random_seed)
    search = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=cfg.n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_holdout(best_search: GridSearchCV, X_test: np.ndarray, y_test: np.ndarray, cfg: Config) -> Dict[str, Any]:
    estimator = best_search.best_estimator_
    y_proba = safe_predict_proba(estimator, X_test)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = evaluate_predictions(y_test, y_pred, y_proba)

    ci_auc = bootstrap_metric_ci(
        y_test, y_pred, y_proba,
        lambda yt, yp, pr: roc_auc_score(yt, pr),
        cfg.n_bootstrap, cfg.random_seed,
    )
    metrics["AUC_95CI"] = f"[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]"
    metrics["Best_Params"] = json.dumps(best_search.best_params_, sort_keys=True)
    return metrics


def build_stacking_model(final_estimators: Dict[str, Any], cfg: Config) -> Optional[StackingClassifier]:
    required = ["Logistic Regression", "SVM", "Random Forest"]
    if not all(k in final_estimators for k in required):
        return None
    estimators = [
        ("lr", clone(final_estimators["Logistic Regression"].best_estimator_)),
        ("svm", clone(final_estimators["SVM"].best_estimator_)),
        ("rf", clone(final_estimators["Random Forest"].best_estimator_)),
    ]
    if "XGBoost" in final_estimators:
        estimators.append(("xgb", clone(final_estimators["XGBoost"].best_estimator_)))
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=5000, class_weight="balanced", random_state=cfg.random_seed),
        cv=cfg.inner_cv_folds,
        n_jobs=cfg.n_jobs,
        passthrough=False,
    )


def save_run_metadata(cfg: Config, outdir: Path) -> None:
    metadata = {
        "config": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "packages_note": "For exact package versions, run: pip freeze > requirements-lock.txt",
        "score_weights": WEIGHTS,
        "interpretation_warning": (
            "MS_Research_Score is exploratory. It is not a validated clinical diagnostic score. "
            "Use external validation before making biological or clinical claims."
        ),
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _print_results_table(holdout_df: pd.DataFrame) -> None:
    display_df = holdout_df.copy()
    for col in display_df.columns:
        if col not in ["Model", "AUC_95CI", "Best_Params", "Balanced_Accuracy"] and display_df[col].dtype in ["float64", "float32"]:
            display_df[col] = display_df[col].round(3)

    display_df = display_df.drop(columns=[c for c in ["Best_Params", "Balanced_Accuracy"] if c in display_df.columns])

    cols = list(display_df.columns)
    if "Model" in cols and "MS_Research_Score" in cols:
        cols.remove("Model")
        cols.remove("MS_Research_Score")
        display_df = display_df[["Model", "MS_Research_Score"] + cols]

    headers = list(display_df.columns)
    widths = [max(len(str(h)), display_df[h].astype(str).str.len().max()) for h in headers]

    print("\n" + "  ".join(_color(f"{headers[i]:<{widths[i]}}", "91") for i in range(len(headers))))
    for _, row in display_df.iterrows():
        model = _color(f"{row['Model']:<{widths[0]}}", "93")
        values = "  ".join(f"{row.iloc[i]:<{widths[i]}}" for i in range(1, len(headers)))
        print(f"{model}  {values}")


def run(cfg: Config) -> None:
    print(render_banner())

    np.random.seed(cfg.random_seed)
    outdir = Path(cfg.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        save_run_metadata(cfg, outdir)
        df = load_geo_series_matrix(cfg.data_file)
        X, y, feature_names = get_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_seed
        )
        registry = model_grid_registry(cfg)

        if cfg.external_models:
            ext_to_tune, ext_pretrained = load_external_models(cfg.external_models, cfg)
        else:
            ext_to_tune, ext_pretrained = {}, {}

        for name, (estimator, grid) in ext_to_tune.items():
            pipeline = estimator if hasattr(estimator, "named_steps") else make_preprocessing_pipeline(estimator, cfg)
            registry[name] = (pipeline, grid or {})

        nested_summaries: List[Dict[str, Any]] = []
        fold_tables: List[pd.DataFrame] = []
        final_searches: Dict[str, GridSearchCV] = {}
        holdout_rows: List[Dict[str, Any]] = []
        holdout_predictions: Dict[str, Dict[str, np.ndarray]] = {}

        total = len(registry)
        print(_color("\nTraining and evaluating models...", "93"), flush=True)
        for i, (name, (pipeline, grid)) in enumerate(registry.items(), 1):
            print(_color(f"\n[{i}/{total}]", "96") + f" Training: {name}", flush=True)
            summary, folds = nested_cv_evaluate_model(name, pipeline, grid, X_train, y_train, cfg)
            nested_summaries.append(summary)
            fold_tables.append(folds)
            print(f"  {_color('OK', '92')} Nested CV complete. Score: {summary['MS_Research_Score']:.3f}", flush=True)

            search = fit_final_model(pipeline, grid, X_train, y_train, cfg)
            final_searches[name] = search
            holdout = evaluate_holdout(search, X_test, y_test, cfg)
            holdout["Model"] = name
            holdout_rows.append(holdout)
            y_proba = safe_predict_proba(search.best_estimator_, X_test)
            holdout_predictions[name] = {
                "y_true": y_test.copy(),
                "y_proba": y_proba.copy(),
                "y_pred": (y_proba >= 0.5).astype(int),
            }
            print(f"  {_color('OK', '92')} Holdout AUC: {holdout['AUC_ROC']:.3f}", flush=True)

        for name, est in ext_pretrained.items():
            y_proba = safe_predict_proba(est, X_test)
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = evaluate_predictions(y_test, y_pred, y_proba)
            metrics["Model"] = name
            metrics["AUC_95CI"] = "not computed"
            metrics["Best_Params"] = "pretrained external model (no tuning)"
            holdout_rows.append(metrics)
            holdout_predictions[name] = {"y_true": y_test.copy(), "y_proba": y_proba.copy(), "y_pred": y_pred.copy()}

        stack = build_stacking_model(final_searches, cfg)
        if stack:
            print(_color("\nTraining Stacking Ensemble...", "93"), flush=True)
            stack.fit(X_train, y_train)
            y_proba = safe_predict_proba(stack, X_test)
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = evaluate_predictions(y_test, y_pred, y_proba)
            metrics["Model"] = "Stacking Ensemble"
            ci_auc = bootstrap_metric_ci(
                y_test, y_pred, y_proba,
                lambda yt, yp, pr: roc_auc_score(yt, pr),
                cfg.n_bootstrap, cfg.random_seed,
            )
            metrics["AUC_95CI"] = f"[{ci_auc[0]:.3f}, {ci_auc[1]:.3f}]"
            metrics["Best_Params"] = "base models already tuned on training set"
            holdout_rows.append(metrics)
            holdout_predictions["Stacking Ensemble"] = {
                "y_true": y_test.copy(), "y_proba": y_proba.copy(), "y_pred": y_pred.copy(),
            }

        nested_df = pd.DataFrame(nested_summaries).sort_values("MS_Research_Score", ascending=False)
        folds_df = pd.concat(fold_tables, ignore_index=True) if fold_tables else pd.DataFrame()
        holdout_df = pd.DataFrame(holdout_rows).sort_values("MS_Research_Score", ascending=False)
        nested_df.to_csv(outdir / "nested_cv_summary.csv", index=False)
        folds_df.to_csv(outdir / "nested_cv_fold_results.csv", index=False)
        holdout_df.to_csv(outdir / "holdout_test_results.csv", index=False)

        pred_rows = []
        for name, pred in holdout_predictions.items():
            for i, (yt, yp, pr) in enumerate(zip(pred["y_true"], pred["y_pred"], pred["y_proba"])):
                pred_rows.append({
                    "Model": name,
                    "Sample_Index_In_Holdout": i,
                    "True_Label": yt,
                    "Predicted_Label": yp,
                    "Predicted_Probability_MS": pr,
                })
        pd.DataFrame(pred_rows).to_csv(outdir / "holdout_predictions.csv", index=False)

        save_leaderboard_plot(holdout_df, outdir)
        save_metric_heatmap(holdout_df, outdir)
        save_roc_plot(holdout_predictions, outdir)
    except Exception:
        error_log = outdir / "benchmark_error.log"
        with open(error_log, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        print(f"\nBenchmark failed. Full traceback written to {error_log}", file=sys.stderr)
        raise

    print("\n" + "=" * 100)
    print(_color("  BENCHMARK RESULTS", "33"))
    print("=" * 100)
    _print_results_table(holdout_df)
    print("\n" + "=" * 100)
    print(f"\nSaved all outputs to: {outdir.resolve()}")
    print("\nReporting note: describe this as 'MS classification from blood RNA profiles', not validated clinical diagnosis.")
