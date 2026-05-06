# ================================================================
# MS DIAGNOSTIC BENCHMARK TOOL
# ================================================================
# Author: Adam Simson
# Affiliation: Synthica Research Group
# Paper: "Evaluating Machine Learning Algorithms for Blood RNA-Based
#         Multiple Sclerosis Diagnosis: A Systematic Benchmark with
#         a Novel Weighted Scoring System"
#
# DESCRIPTION:
# This tool allows researchers to test their own machine learning
# models against the benchmark established in the paper above.
# Simply add your model in the section marked below, run the script,
# and see how your model compares to the published benchmark.
#
# REQUIREMENTS:
# pip install pandas numpy scikit-learn imbalanced-learn matplotlib
#
# DATA REQUIRED:
# Download GSE17048_series_matrix.txt.gz from:
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17048
# Place it in the same directory as this script.
#
# BENCHMARK LEADERBOARD (Adam Simson, 2025):
# 1. Stacking Ensemble   — MS Score: 97.85
# 2. SVM                 — MS Score: 97.55
# 3. Neural Network      — MS Score: 97.42
# 4. Logistic Regression — MS Score: 96.42
# 5. Random Forest       — MS Score: 93.92
# 6. XGBoost             — MS Score: 89.35
# 7. Gradient Boosting   — MS Score: 87.68
# 8. Naive Bayes         — MS Score: 75.21
# 9. KNN                 — MS Score: 73.84
# ================================================================

import pandas as pd
import numpy as np
import gzip
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# SCORING SYSTEM
# Weights reflect clinical importance of each metric
# ================================================================
WEIGHTS = {
    "AUC-ROC":   0.30,   # Most important — overall discrimination
    "Recall":    0.25,   # Critical — missing MS is dangerous
    "F1":        0.20,   # Balance of precision and recall
    "Precision": 0.15,   # Correct positive predictions
    "Brier":     0.10,   # Probability calibration quality
}

# Published benchmark scores
BENCHMARK = {
    "Stacking Ensemble":   97.85,
    "SVM":                 97.55,
    "Neural Network":      97.42,
    "Logistic Regression": 96.42,
    "Random Forest":       93.92,
    "XGBoost":             89.35,
    "Gradient Boosting":   87.68,
    "Naive Bayes":         75.21,
    "KNN":                 73.84,
}

def compute_ms_score(auc, recall, f1, precision, brier):
    """
    Compute the MS Diagnostic Score out of 100.
    
    Parameters:
        auc       : float — AUC-ROC score (0 to 1)
        recall    : float — Recall as percentage (0 to 100)
        f1        : float — F1 Score as percentage (0 to 100)
        precision : float — Precision as percentage (0 to 100)
        brier     : float — Brier Score (0 to 1, lower is better)
    
    Returns:
        float — MS Diagnostic Score out of 100
    """
    brier_component = max(0, (1 - brier) * 100)
    score = (
        auc * 100   * WEIGHTS["AUC-ROC"]   +
        recall      * WEIGHTS["Recall"]     +
        f1          * WEIGHTS["F1"]         +
        precision   * WEIGHTS["Precision"]  +
        brier_component * WEIGHTS["Brier"]
    )
    return round(score, 2)


# ================================================================
# DATA LOADING & PREPROCESSING
# Do not modify this section
# ================================================================
def load_and_prepare_data():
    print("Loading GSE17048 dataset...")
    with gzip.open("GSE17048_series_matrix.txt.gz", "rt") as f:
        lines = f.readlines()

    label_line = lines[33]
    labels = [l.strip('"') for l in label_line.strip().split("\t")[1:]]
    conditions = []
    for label in labels:
        if "healthy" in label.lower(): conditions.append("HC")
        elif "RR" in label: conditions.append("RR")
        elif "PP" in label: conditions.append("PP")
        elif "SP" in label: conditions.append("SP")
        else: conditions.append("Other")

    data_start = 0
    for i, line in enumerate(lines):
        if "!series_matrix_table_begin" in line:
            data_start = i + 1
            break

    data_rows, gene_ids = [], []
    for line in lines[data_start:]:
        if "!series_matrix_table_end" in line: break
        if line.startswith('"ID_REF"'): continue
        parts = line.strip().split("\t")
        gene_id = parts[0].strip('"')
        try:
            values = [float(x) for x in parts[1:]]
            if len(values) == len(conditions):
                gene_ids.append(gene_id)
                data_rows.append(values)
        except:
            continue

    df = pd.DataFrame(data_rows, index=gene_ids,
                      columns=conditions).T.reset_index()
    df = df.rename(columns={"index": "Condition"})
    df["Label"] = (df["Condition"] != "HC").astype(int)

    # Feature selection — variance filter + RFE
    print("Running feature selection (this may take a few minutes)...")
    gene_cols = [c for c in df.columns if c not in ["Condition", "Label"]]
    variances = df[gene_cols].var()
    top_1000 = variances.nlargest(1000).index.tolist()

    scaler_rfe = StandardScaler()
    X_rfe = scaler_rfe.fit_transform(df[top_1000].values)
    y_rfe = df["Label"].values

    rfe = RFE(
        RandomForestClassifier(n_estimators=50, random_state=42),
        n_features_to_select=200, step=50
    )
    rfe.fit(X_rfe, y_rfe)
    selected = [top_1000[i] for i, s in enumerate(rfe.support_) if s]
    print(f"Selected {len(selected)} genes")

    # Scale and balance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected].values)
    y = df["Label"].values

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2,
        random_state=42, stratify=y_res
    )

    print(f"Data ready: {len(selected)} genes, "
          f"{len(X_res)} patients after SMOTE")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

    return X_train, X_test, y_train, y_test


def evaluate_and_compare(model_name, model, X_train, X_test,
                          y_train, y_test):
    """
    Evaluate a model and compare it to the published benchmark.
    
    Parameters:
        model_name : str   — name of your model
        model      : object — sklearn-compatible model
        X_train    : array — training features
        X_test     : array — test features
        y_train    : array — training labels
        y_test     : array — test labels
    """
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    auc       = round(roc_auc_score(y_test, proba), 4)
    recall    = round(recall_score(y_test, preds) * 100, 2)
    f1        = round(f1_score(y_test, preds) * 100, 2)
    precision = round(precision_score(y_test, preds) * 100, 2)
    accuracy  = round(accuracy_score(y_test, preds) * 100, 2)
    brier     = round(brier_score_loss(y_test, proba), 4)
    ms_score  = compute_ms_score(auc, recall, f1, precision, brier)

    # Print results
    print("\n" + "=" * 50)
    print(f"RESULTS — {model_name}")
    print("=" * 50)
    print(f"  AUC-ROC:    {auc}")
    print(f"  Recall:     {recall}%")
    print(f"  F1 Score:   {f1}%")
    print(f"  Precision:  {precision}%")
    print(f"  Accuracy:   {accuracy}%")
    print(f"  Brier:      {brier}")
    print(f"  MS SCORE:   {ms_score} / 100")

    # Rank against benchmark
    all_scores = dict(BENCHMARK)
    all_scores[model_name] = ms_score
    sorted_scores = sorted(all_scores.items(),
                           key=lambda x: x[1], reverse=True)
    rank = [i + 1 for i, (n, _) in enumerate(sorted_scores)
            if n == model_name][0]
    total = len(sorted_scores)

    print(f"\n  RANK: {rank} of {total}")
    if rank == 1:
        print("  NEW BENCHMARK RECORD!")
    elif rank <= 3:
        print("  Top 3 — Excellent!")
    elif rank <= 5:
        print("  Top 5 — Strong result!")
    else:
        print("  Keep optimising!")

    # Plot leaderboard
    plt.figure(figsize=(12, 7))
    names  = [n for n, _ in sorted_scores]
    scores = [s for _, s in sorted_scores]
    colors = ["#e74c3c" if n == model_name
              else "#3498db" for n in names]
    bars = plt.barh(names[::-1], scores[::-1],
                    color=colors[::-1], alpha=0.85)
    plt.axvline(x=90, color="green", linestyle="--",
                linewidth=1.5, label="Excellent (90+)")
    plt.axvline(x=80, color="orange", linestyle="--",
                linewidth=1.5, label="Good (80+)")
    plt.axvline(x=70, color="red", linestyle="--",
                linewidth=1.5, label="Acceptable (70+)")
    for bar, score in zip(bars[::-1], scores):
        plt.text(bar.get_width() + 0.3,
                 bar.get_y() + bar.get_height() / 2,
                 f"{score}", va="center",
                 fontsize=9, fontweight="bold")
    plt.xlabel("MS Diagnostic Score (out of 100)", fontsize=11)
    plt.title(
        f"Your Model vs Benchmark\n"
        f"{model_name} ranked {rank} of {total}",
        fontsize=11
    )
    plt.legend(fontsize=9)
    plt.xlim(50, 115)
    plt.tight_layout()
    plt.savefig("my_benchmark_result.png", dpi=300)
    plt.show()
    print("  Graph saved as my_benchmark_result.png")

    return ms_score


# ================================================================
# ADD YOUR MODEL HERE
# ================================================================
# Instructions:
# 1. Import your model at the top of this section
# 2. Set YOUR_MODEL_NAME to your model's name
# 3. Set YOUR_MODEL to your model instance
# 4. Run the script
#
# Example:
#   from sklearn.ensemble import RandomForestClassifier
#   YOUR_MODEL_NAME = "My Random Forest"
#   YOUR_MODEL = RandomForestClassifier(n_estimators=500)
#
# Your model must be sklearn-compatible, meaning it must have
# fit(), predict() and predict_proba() methods.
# ================================================================

from sklearn.ensemble import RandomForestClassifier   # change this

YOUR_MODEL_NAME = "My Custom Model"                   # change this
YOUR_MODEL = RandomForestClassifier(                  # change this
    n_estimators=100,
    random_state=42
)

# ================================================================
# RUN — do not modify below this line
# ================================================================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    evaluate_and_compare(
        YOUR_MODEL_NAME, YOUR_MODEL,
        X_train, X_test, y_train, y_test
    )