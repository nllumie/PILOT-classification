import argparse
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    make_classification,
    make_moons,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ucimlrepo import fetch_ucirepo

from pilot_classification import PILOTClassifier, RaFFLEClassifier


SEEDS = [42, 52, 62]
CV_SEED = 42
OUT = pathlib.Path("experiments/classification/results")
DATASET_NAMES = [
    "synthetic_linear",
    "synthetic_moons",
    "breast_cancer",
    "banknote",
    "ionosphere",
    "pima",
]



def load_datasets(seed, selected=None):
    """Load requested datasets with the positive class encoded as 1."""

    if selected is None:
        selected = DATASET_NAMES

    data = {}

    # Synthetic linear dataset.
    if "synthetic_linear" in selected:
        X, y = make_classification(
            n_samples=600,
            n_features=6,
            n_informative=3,
            n_redundant=0,
            flip_y=0.05,
            class_sep=1.2,
            random_state=seed,
        )
        data["synthetic_linear"] = (X, y)

    # Synthetic nonlinear dataset.
    if "synthetic_moons" in selected:
        X, y = make_moons(
            n_samples=600,
            noise=0.25,
            random_state=seed,
        )
        data["synthetic_moons"] = (X, y)

    # Breast Cancer Wisconsin.
    if "breast_cancer" in selected:
        bc = load_breast_cancer()
        # Positive class = malignant (original sklearn label 0).
        data["breast_cancer"] = (
            bc.data,
            (bc.target == 0).astype(int),
        )

    # UCI Banknote Authentication, ID 267.
    if "banknote" in selected:
        banknote = fetch_ucirepo(id=267)
        X = banknote.data.features.to_numpy(dtype=float)
        y = pd.to_numeric(
            banknote.data.targets.iloc[:, 0]
        ).to_numpy(dtype=int)

        # Positive class = forged (label 1).
        data["banknote"] = (X, y)

    # UCI Ionosphere, ID 52.
    if "ionosphere" in selected:
        ionosphere = fetch_ucirepo(id=52)
        X = ionosphere.data.features.to_numpy(dtype=float)

        labels = (
            ionosphere.data.targets.iloc[:, 0]
            .astype(str)
            .str.strip()
            .str.lower()
            .to_numpy()
        )

        # Positive class = good (label "g"); negative class = bad ("b").
        if not np.isin(labels, ["g", "b"]).all():
            raise ValueError(
                f"Unexpected Ionosphere target labels: {np.unique(labels)}"
            )
        y = (labels == "g").astype(int)
        data["ionosphere"] = (X, y)

    # Pima Indians Diabetes, OpenML ID 37.
    if "pima" in selected:
        pima = fetch_openml(data_id=37, as_frame=True)
        X = pima.data.to_numpy(dtype=float)

        labels = (
            pima.target.astype(str)
            .str.strip()
            .str.lower()
            .to_numpy()
        )

        # Positive class = diabetic ("tested_positive").
        expected_labels = {"tested_negative", "tested_positive"}
        if set(np.unique(labels)) != expected_labels:
            raise ValueError(
                f"Unexpected Pima target labels: {np.unique(labels)}"
            )
        y = (labels == "tested_positive").astype(int)

        # Zero represents missing data in these five features:
        # glucose, blood pressure, skin thickness, insulin, and BMI.
        X = X.copy()
        missing_columns = [1, 2, 3, 4, 5]

        for column in missing_columns:
            X[X[:, column] == 0, column] = np.nan

        data["pima"] = (X, y)

    return data


def make_models(seed):
    """Create the same six models with fixed settings."""
    return {
        "Majority": DummyClassifier(
            strategy="most_frequent"
        ),
        "LogReg": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "DecisionTree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=10,
                random_state=seed,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=5,
                random_state=seed,
            )),
        ]),
        "PILOT": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", PILOTClassifier(
                max_depth=4,
                min_samples_leaf=10,
                random_state=seed,
            )),
        ]),
        "RaFFLE": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RaFFLEClassifier(
                n_estimators=30,
                max_depth=4,
                min_samples_leaf=10,
                random_state=seed,
            )),
        ]),
    }


def calculate_metrics(model, X_test, y_test):
    """Calculate all required classification metrics."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    classes = list(model.classes_)
    positive_col = classes.index(1)
    p_pos = proba[:, positive_col]

    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(
            y_test, pred, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            y_test, pred, pos_label=1, zero_division=0
        ),
        "f1_positive": f1_score(
            y_test, pred, pos_label=1, zero_division=0
        ),
        "macro_f1": f1_score(
            y_test, pred, average="macro"
        ),
        "roc_auc": roc_auc_score(y_test, p_pos),
        "log_loss": log_loss(
            y_test,
            np.column_stack([1 - p_pos, p_pos]),
            labels=[0, 1],
        ),
    }


def count_discarded_candidates(model):
    """Count discarded logistic candidates across RaFFLE trees."""
    raffle = model.named_steps["model"]

    return sum(
        getattr(estimator, "_nonconverged_candidates", 0)
        for estimator in raffle.estimators_
    )


def run_train_test(ds_name, X, y, seeds):
    """Run stratified 80/20 train-test evaluation for each seed."""
    all_rows = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=seed,
        )

        print(
            f"\nTrain-test: {ds_name}, seed={seed}",
            flush=True,
        )

        for model_name, model in make_models(seed).items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start

            metrics = calculate_metrics(model, X_test, y_test)

            discarded = 0
            if model_name == "RaFFLE":
                discarded = count_discarded_candidates(model)
                print(
                    f"RaFFLE discarded candidates: {discarded}",
                    flush=True,
                )

            row = {
                "dataset": ds_name,
                "seed": seed,
                "model": model_name,
                **metrics,
                "fit_seconds": fit_seconds,
                "nonconverged_candidates": discarded,
            }
            all_rows.append(row)

            print(
                f"{model_name:12s} "
                f"acc={metrics['accuracy']:.3f} "
                f"precision={metrics['precision']:.3f} "
                f"recall={metrics['recall']:.3f} "
                f"f1={metrics['f1_positive']:.3f} "
                f"auc={metrics['roc_auc']:.3f} "
                f"t={fit_seconds:.2f}s",
                flush=True,
            )

    results = pd.DataFrame(all_rows)

    for seed in seeds:
        seed_results = results[results["seed"] == seed]
        path = OUT / f"{ds_name}_split_seed{seed}.csv"
        seed_results.to_csv(path, index=False)
        print("Saved:", path)

    return results


def run_cross_validation(ds_name, X, y):
    """Run one 5-fold stratified CV evaluation per dataset."""
    seed = CV_SEED

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    rows = []

    print(
        f"\n5-fold cross-validation: {ds_name}",
        flush=True,
    )

    for model_name, base_model in make_models(seed).items():
        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y), start=1
        ):
            model = clone(base_model)

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start

            metrics = calculate_metrics(model, X_test, y_test)

            discarded = 0
            if model_name == "RaFFLE":
                discarded = count_discarded_candidates(model)

            rows.append({
                "dataset": ds_name,
                "model": model_name,
                "fold": fold,
                **metrics,
                "fit_seconds": fit_seconds,
                "nonconverged_candidates": discarded,
            })

            print(
                f"{model_name:12s} fold={fold} "
                f"acc={metrics['accuracy']:.3f} "
                f"auc={metrics['roc_auc']:.3f} "
                f"t={fit_seconds:.2f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    path = OUT / f"{ds_name}_cv.csv"
    results.to_csv(path, index=False)

    print("Saved:", path)

    return results


def make_summary(split_results, cv_results):
    """Create mean and standard deviation comparison tables."""
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_positive",
        "macro_f1",
        "roc_auc",
        "log_loss",
    ]

    if not split_results.empty:
        split_summary = (
            split_results.groupby(["dataset", "model"])[metrics]
            .agg(["mean", "std"])
            .round(3)
        )
        split_summary.to_csv(OUT / "split_comparison.csv")

    if not cv_results.empty:
        cv_summary = (
            cv_results.groupby(["dataset", "model"])[metrics]
            .agg(["mean", "std"])
            .round(3)
        )
        cv_summary.to_csv(OUT / "cv_comparison.csv")


def make_roc_auc_chart(split_results):
    """Create a ROC-AUC bar chart with standard deviation error bars."""
    if split_results.empty:
        return

    summary = (
        split_results.groupby(["dataset", "model"])["roc_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    datasets = summary["dataset"].unique()
    models = list(make_models(CV_SEED).keys())

    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(11, max(4, 4 * len(datasets))),
        squeeze=False,
    )

    for ax, dataset in zip(axes[:, 0], datasets):
        subset = summary[summary["dataset"] == dataset]
        subset = subset.set_index("model").reindex(models)

        ax.bar(
            models,
            subset["mean"],
            yerr=subset["std"].fillna(0),
            capsize=4,
        )
        ax.set_title(dataset)
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    path = OUT / "roc_auc_by_dataset.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print("Saved chart:", path)


def load_all_results():
    """Combine saved main-benchmark CSVs and skip outdated result files."""
    required_columns = {
        "dataset",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_positive",
        "macro_f1",
        "roc_auc",
        "log_loss",
    }

    split_frames = []
    cv_frames = []

    for dataset in DATASET_NAMES:
        # Each dataset has one train-test CSV per seed.
        for path in sorted(OUT.glob(f"{dataset}_split_seed*.csv")):
            frame = pd.read_csv(path)
            missing = required_columns.difference(frame.columns)

            if missing:
                print(
                    f"Skipping outdated file {path}; missing columns: "
                    f"{', '.join(sorted(missing))}",
                    flush=True,
                )
                continue

            split_frames.append(frame)

        # Each dataset has one cross-validation CSV.
        cv_path = OUT / f"{dataset}_cv.csv"
        if cv_path.exists():
            frame = pd.read_csv(cv_path)
            missing = required_columns.difference(frame.columns)

            if missing:
                print(
                    f"Skipping outdated file {cv_path}; missing columns: "
                    f"{', '.join(sorted(missing))}",
                    flush=True,
                )
            else:
                cv_frames.append(frame)

    split = (
        pd.concat(split_frames, ignore_index=True)
        if split_frames
        else pd.DataFrame()
    )
    cv = (
        pd.concat(cv_frames, ignore_index=True)
        if cv_frames
        else pd.DataFrame()
    )

    return split, cv

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use seed 42 only for train-test evaluation",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASET_NAMES,
        help="Run one or more datasets",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only rebuild comparison tables and chart from saved CSVs",
    )

    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        split_combined, cv_combined = load_all_results()

        if split_combined.empty and cv_combined.empty:
            raise RuntimeError(
                f"No saved benchmark CSV files were found in {OUT}. "
                "Run at least one dataset before using --summary-only."
            )

        make_summary(split_combined, cv_combined)
        make_roc_auc_chart(split_combined)
        print("Summary rebuilt from", OUT)
        return

    seeds = SEEDS[:1] if args.quick else SEEDS

    # Load only the requested datasets, or all six if --dataset was omitted.
    # Synthetic datasets are generated with seed 42 for consistent data.
    datasets = load_datasets(42, selected=args.dataset)

    for ds_name, (X, y) in datasets.items():
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        print(
            f"\nDataset: {ds_name} | "
            f"Rows: {len(y)} | Features: {X.shape[1]} | "
            f"Positive: {int(y.sum())} | "
            f"Negative: {int(len(y) - y.sum())}",
            flush=True,
        )

        split_results = run_train_test(ds_name, X, y, seeds)
        run_cross_validation(ds_name, X, y)

    # Rebuild summaries from every saved main-benchmark CSV, not just
    # the datasets included in this invocation.
    split_combined, cv_combined = load_all_results()
    make_summary(split_combined, cv_combined)
    make_roc_auc_chart(split_combined)

    print("\nFinished.")
    print("Results folder:", OUT)


if __name__ == "__main__":
    main()
