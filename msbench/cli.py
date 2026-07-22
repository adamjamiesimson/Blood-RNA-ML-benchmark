"""Command-line entry point for the MS blood RNA ML benchmark."""

from __future__ import annotations

import argparse
import sys

from .config import Config


def parse_args(argv=None) -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(
        prog="msbench",
        description="Benchmark ML models for MS classification from blood RNA (GSE17048).",
    )
    parser.add_argument("--data-file", default=defaults.data_file, help="Path to GSE17048_series_matrix.txt.gz")
    parser.add_argument("--results-dir", default=defaults.results_dir, help="Output directory")
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--outer-cv-folds", type=int, default=defaults.outer_cv_folds)
    parser.add_argument("--inner-cv-folds", type=int, default=defaults.inner_cv_folds)
    parser.add_argument("--n-bootstrap", type=int, default=defaults.n_bootstrap)
    parser.add_argument("--top-variance", type=int, default=defaults.top_variance)
    parser.add_argument("--k-best-features", type=int, default=defaults.k_best_features)
    parser.add_argument("--n-jobs", type=int, default=defaults.n_jobs)
    parser.add_argument(
        "--external-models",
        default=None,
        help="External models: a .py module defining EXTERNAL_MODELS, or a dir/file of .joblib/.pkl",
    )
    args = parser.parse_args(argv)
    return Config(
        data_file=args.data_file,
        results_dir=args.results_dir,
        test_size=args.test_size,
        outer_cv_folds=args.outer_cv_folds,
        inner_cv_folds=args.inner_cv_folds,
        n_bootstrap=args.n_bootstrap,
        top_variance=args.top_variance,
        k_best_features=args.k_best_features,
        n_jobs=args.n_jobs,
        external_models=args.external_models,
    )


def main(argv=None) -> None:
    cfg = parse_args(argv)
    try:
        from .benchmark import run
    except ImportError as e:
        print(
            f"Missing dependency: {e.name}. Install requirements first:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)
    run(cfg)


if __name__ == "__main__":
    main()
