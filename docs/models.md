# Models

The benchmark ships with nine estimators. Eight are tuned with nested cross-validation; the ninth is a
stacking ensemble built from the tuned base models. They give any custom model a fixed, comparable baseline —
and let you run the benchmark end to end without writing a model at all.

Every estimator runs inside the same leakage-safe pipeline
([`msbench/models.py`](../msbench/models.py) · `make_preprocessing_pipeline`):

```
median imputation → top-variance filter → standardize → SelectKBest → SMOTE (train only) → estimator
```

## Tuned panel

| Model | Key hyperparameters searched |
|-------|------------------------------|
| Logistic Regression | `C`, L2 penalty, `kbest__k` |
| SVM | `C`, kernel (linear/rbf), `kbest__k` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |
| Neural Network (MLP) | hidden layer sizes, `alpha`, learning-rate schedule |
| Naive Bayes | `var_smoothing` |
| KNN | `n_neighbors`, weighting |

The full grids live in `model_grid_registry` in [`msbench/models.py`](../msbench/models.py).

XGBoost is optional — if it (or, on macOS, `libomp`) is unavailable, the benchmark skips it and continues.

## Stacking Ensemble

A `StackingClassifier` over the tuned Logistic Regression, SVM, and Random Forest (plus XGBoost when
available), with logistic regression as the meta-learner. It is evaluated on the holdout set only.

## Adding your own

You do not edit this registry. Pass a model file instead:

```bash
python run_benchmark.py --external-models examples/external_models.py
```

See [`examples/external_models.py`](../examples/external_models.py) for the format.
