"""Run four-way split conformal evaluations for the Step 5 classifiers."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.classification.conformal_sets import SplitConformalClassifier
from experiments.classification.run_classification import (
    SEEDS,
    load_datasets,
    make_models,
)


DATASET_NAMES = ("breast_cancer", "banknote", "pima")
MODEL_NAMES = ("RaFFLE", "RandomForest", "LogReg")
ALPHA = 0.10
OUTPUT_DIR = pathlib.Path("experiments/classification/outputs")
PARTITION_NAMES = ("train", "validation", "calibration", "test")


def _stratified_split(
    indices: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    try:
        left, right = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y[indices],
        )
        return left, right, True
    except ValueError:
        left, right = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
        return left, right, False


def make_four_way_split(
    y: object,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    """Split original row indices into approximate 50/15/15/20 partitions.

    The staged proportions reserve 20% for test, 30% of the full data for
    validation plus calibration, then split that reserve equally. Scikit-learn
    rounds partition sizes to whole rows; stratification is attempted at each
    stage and used when feasible.
    """
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if labels.size < 4:
        raise ValueError("at least four rows are required for four-way splitting")
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise ValueError("y must contain finite labels")

    indices = np.arange(labels.size, dtype=np.int64)
    train_validation_calibration, test_indices, test_stratified = (
        _stratified_split(indices, labels, 0.20, seed)
    )
    train_indices, validation_calibration, holdout_stratified = (
        _stratified_split(
            train_validation_calibration,
            labels,
            0.375,
            seed + 1,
        )
    )
    validation_indices, calibration_indices, final_stratified = (
        _stratified_split(
            validation_calibration,
            labels,
            0.50,
            seed + 2,
        )
    )

    partitions = {
        "train": np.asarray(train_indices, dtype=np.int64),
        "validation": np.asarray(validation_indices, dtype=np.int64),
        "calibration": np.asarray(calibration_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }
    stratified = {
        "test": test_stratified,
        "validation_calibration": holdout_stratified,
        "validation_vs_calibration": final_stratified,
    }

    flattened = np.concatenate(tuple(partitions.values()))
    if (
        flattened.size != labels.size
        or np.unique(flattened).size != labels.size
        or not np.array_equal(np.sort(flattened), indices)
    ):
        raise RuntimeError("four-way split must be disjoint and cover every row")

    return partitions, stratified


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _class_column_name(label: Any) -> str:
    return f"probability_class_{_python_scalar(label)}"


def _print_partition_counts(
    dataset: str,
    seed: int,
    y: np.ndarray,
    partitions: dict[str, np.ndarray],
) -> None:
    classes = np.unique(y)
    print(f"\n{dataset}, seed={seed} partition class counts:")
    for partition_name in PARTITION_NAMES:
        counts = Counter(y[partitions[partition_name]].tolist())
        display_counts = {
            str(_python_scalar(label)): counts[label] for label in classes
        }
        print(f"  {partition_name}: {display_counts}")


def _save_split_indices(
    dataset: str,
    seed: int,
    partitions: dict[str, np.ndarray],
    output_dir: pathlib.Path,
) -> pathlib.Path:
    rows = [
        {
            "dataset": dataset,
            "seed": seed,
            "partition": partition_name,
            "original_index": int(index),
        }
        for partition_name in PARTITION_NAMES
        for index in partitions[partition_name]
    ]
    path = output_dir / f"split_indices_{dataset}_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _calculate_summary(
    dataset: str,
    seed: int,
    model_name: str,
    y_test: np.ndarray,
    prediction_sets: list[list[Any]],
    classes: np.ndarray,
) -> dict[str, Any]:
    set_sizes = np.asarray([len(prediction_set) for prediction_set in prediction_sets])
    covered = np.asarray(
        [
            any(label == candidate for candidate in prediction_set)
            for label, prediction_set in zip(y_test, prediction_sets)
        ],
        dtype=bool,
    )
    summary: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "model": model_name,
        "test_count": int(y_test.size),
        "overall_coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(set_sizes)),
        "empty_set_rate": float(np.mean(set_sizes == 0)),
        "both_label_rate": float(np.mean(set_sizes == classes.size)),
    }
    for label in classes:
        class_mask = y_test == label
        suffix = _python_scalar(label)
        summary[f"test_count_class_{suffix}"] = int(np.sum(class_mask))
        summary[f"coverage_class_{suffix}"] = (
            float(np.mean(covered[class_mask]))
            if np.any(class_mask)
            else math.nan
        )
    return summary


def evaluate_dataset(
    dataset: str,
    X: object,
    y: object,
    seed: int,
    output_dir: pathlib.Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit, calibrate, and evaluate all configured models on one dataset/seed."""
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("X must be two-dimensional and match y's row count")
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")

    partitions, stratified = make_four_way_split(labels, seed)
    _print_partition_counts(dataset, seed, labels, partitions)
    print(f"  stratification used: {stratified}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = _save_split_indices(dataset, seed, partitions, output_dir)
    print("Saved:", split_path)

    X_train, y_train = features[partitions["train"]], labels[partitions["train"]]
    X_cal, y_cal = features[partitions["calibration"]], labels[partitions["calibration"]]
    X_test, y_test = features[partitions["test"]], labels[partitions["test"]]

    available_models = make_models(seed)
    model_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        model = available_models[model_name]
        model.fit(X_train, y_train)
        classes = np.asarray(model.classes_)

        conformal = SplitConformalClassifier(model, alpha=ALPHA)
        conformal.calibrate(X_cal, y_cal)
        predictions = model.predict(X_test)
        probabilities = np.asarray(model.predict_proba(X_test))
        prediction_sets = conformal.predict_sets(X_test)

        if probabilities.shape != (y_test.size, classes.size):
            raise ValueError(
                f"{model_name} returned probabilities with an unexpected shape"
            )
        if len(prediction_sets) != y_test.size:
            raise ValueError(
                f"{model_name} returned an unexpected number of prediction sets"
            )

        example_rows: list[dict[str, Any]] = []
        for row_index, (
            true_label,
            predicted_label,
            row_probabilities,
            prediction_set,
        ) in enumerate(zip(y_test, predictions, probabilities, prediction_sets)):
            result: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
                "original_index": int(partitions["test"][row_index]),
                "true_label": _python_scalar(true_label),
                "predicted_label": _python_scalar(predicted_label),
                "conformal_set": json.dumps(
                    [_python_scalar(label) for label in prediction_set]
                ),
                "set_size": len(prediction_set),
                "covered": any(
                    true_label == candidate for candidate in prediction_set
                ),
            }
            result.update(
                {
                    _class_column_name(label): float(row_probabilities[column])
                    for column, label in enumerate(classes)
                }
            )
            example_rows.append(result)

        per_model = pd.DataFrame(example_rows)
        model_rows.append(per_model)
        summary = _calculate_summary(
            dataset, seed, model_name, y_test, prediction_sets, classes
        )
        summary["alpha"] = ALPHA
        summary["calibration_count"] = int(y_cal.size)
        summary["calibration_threshold"] = conformal.threshold_
        summary_rows.append(summary)
        print(
            f"  {model_name}: threshold={conformal.threshold_:.6g}, "
            f"coverage={summary_rows[-1]['overall_coverage']:.3f}, "
            f"mean_set_size={summary_rows[-1]['average_set_size']:.3f}"
        )

    predictions_frame = pd.concat(model_rows, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows)
    predictions_path = output_dir / f"predictions_{dataset}_seed{seed}.csv"
    summary_path = output_dir / f"summary_{dataset}_seed{seed}.csv"
    predictions_frame.to_csv(predictions_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    print("Saved:", predictions_path)
    print("Saved:", summary_path)
    return predictions_frame, summary_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="Datasets to evaluate (default: all supported conformal datasets)",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=SEEDS,
        help=f"Step 5 split seeds (default: {SEEDS})",
    )
    args = parser.parse_args()

    # Load through the existing Step 5 loader; synthetic data generation seed
    # is fixed independently of the conformal partition seeds.
    datasets = load_datasets(42, selected=args.dataset)
    for dataset_name in args.dataset:
        X, y = datasets[dataset_name]
        for seed in args.seed:
            evaluate_dataset(dataset_name, X, y, seed)


if __name__ == "__main__":
    main()
