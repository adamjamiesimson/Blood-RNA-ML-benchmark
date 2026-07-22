"""Load the GSE17048 GEO series matrix into a samples x genes table."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _find_first_line_index(lines: List[str], token: str) -> int:
    for i, line in enumerate(lines):
        if token in line:
            return i
    raise ValueError(f"Could not find required token in GEO matrix: {token}")


def _extract_conditions_from_geo_metadata(lines: List[str], n_samples: int) -> List[str]:
    """Extract condition labels from GEO metadata with robust fallbacks."""
    candidate_tokens = [
        "!Sample_characteristics_ch1",
        "!Sample_title",
        "!Sample_source_name_ch1",
    ]

    metadata_lines = [line for line in lines if any(tok in line for tok in candidate_tokens)]
    sample_strings: List[str] = ["" for _ in range(n_samples)]

    for line in metadata_lines:
        parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")[1:]]
        if len(parts) != n_samples:
            continue
        for i, p in enumerate(parts):
            sample_strings[i] += " " + p

    conditions = []
    for text in sample_strings:
        lower = text.lower()
        if "healthy" in lower or "control" in lower or lower.strip() == "hc":
            conditions.append("HC")
        elif "relapsing" in lower or "rr" in text or "rrms" in lower:
            conditions.append("RRMS")
        elif "primary" in lower or "pp" in text or "ppms" in lower:
            conditions.append("PPMS")
        elif "secondary" in lower or "sp" in text or "spms" in lower:
            conditions.append("SPMS")
        elif "multiple sclerosis" in lower or "ms" in lower:
            conditions.append("MS")
        else:
            conditions.append("Other")

    # Fallback for the known GSE17048 formatting used in the original script.
    if all(c == "Other" for c in conditions):
        try:
            label_line = lines[33]
            labels = [l.strip('"') for l in label_line.strip().split("\t")[1:]]
            conditions = []
            for label in labels:
                lower = label.lower()
                if "healthy" in lower:
                    conditions.append("HC")
                elif "RR" in label:
                    conditions.append("RRMS")
                elif "PP" in label:
                    conditions.append("PPMS")
                elif "SP" in label:
                    conditions.append("SPMS")
                else:
                    conditions.append("Other")
        except Exception:
            pass

    if len(conditions) != n_samples:
        raise ValueError("Could not infer sample conditions reliably from GEO metadata.")
    return conditions


def load_geo_series_matrix(filepath: str | Path) -> pd.DataFrame:
    """Load a GEO series matrix into a samples x genes dataframe.

    Returns columns: Condition, Label, plus expression features.
    Label: 0 = healthy control, 1 = MS case.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            "Download GEO accession GSE17048 and place the .txt.gz file at that path "
            "(the repository ships it under data/)."
        )

    print(f"Loading GEO series matrix: {filepath}")
    opener = gzip.open if filepath.suffix == ".gz" else open
    with opener(filepath, "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    table_start = _find_first_line_index(lines, "!series_matrix_table_begin") + 1
    table_end = _find_first_line_index(lines, "!series_matrix_table_end")

    header = [h.strip().strip('"') for h in lines[table_start].rstrip("\n").split("\t")]
    sample_ids = header[1:]
    n_samples = len(sample_ids)
    conditions = _extract_conditions_from_geo_metadata(lines, n_samples)

    gene_ids: List[str] = []
    data: List[List[float]] = []
    for line in lines[table_start + 1:table_end]:
        parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
        if len(parts) != n_samples + 1:
            continue
        gene_id = parts[0]
        try:
            values = [float(x) if x not in {"", "NA", "NaN"} else np.nan for x in parts[1:]]
        except ValueError:
            continue
        gene_ids.append(gene_id)
        data.append(values)

    if not data:
        raise ValueError("No numeric expression rows were parsed from the GEO matrix.")

    expression = pd.DataFrame(np.array(data).T, columns=gene_ids)
    expression.insert(0, "Condition", conditions)
    expression.insert(1, "Label", [0 if c == "HC" else 1 for c in conditions])

    before = len(expression)
    expression = expression[expression["Condition"] != "Other"].reset_index(drop=True)
    removed = before - len(expression)

    print(f"Parsed {expression.shape[0]} samples x {expression.shape[1] - 2:,} features")
    print(f"Condition counts: {expression['Condition'].value_counts().to_dict()}")
    if removed:
        print(f"Removed {removed} samples with unclear labels.")
    print(f"Binary label counts: {expression['Label'].value_counts().to_dict()}  (0=HC, 1=MS)")
    return expression


def get_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = [c for c in df.columns if c not in {"Condition", "Label"}]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["Label"].to_numpy(dtype=int)
    return X, y, feature_cols
