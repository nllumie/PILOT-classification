
import argparse
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pilot_classification import PILOTClassifier, RaFFLEClassifier

SEEDS = [42, 52, 62]
OUT = pathlib.Path("experiments/classification/results")


def load_datasets(seed):
    """Return {name: (X, y_pos)} where y_pos is 1 for the positive class."""
    data = {}

    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=3, n_redundant=0,
        flip_y=0.05, class_sep=1.2, random_state=seed,
    )
    data["synthetic_linear"] = (X, y)

    X, y = make_moons(n_samples=600, noise=0.25, random_state=seed)
    data["synthetic_moons"] = (X, y)

    bc = load_breast_cancer()
    # Positive class = malignant (label 0 in sklearn), relabelled to 1.
    data["breast_cancer"] = (bc.data, (bc.target == 0).astype(int))

    try:
        from ucimlrepo import fetch_ucirepo
        banknote = fetch_ucirepo(id=267)
        X = banknote.data.features.to_numpy(dtype=float)
        y = banknote.data.targets.iloc[:, 0].to_numpy()
        # Positive class = forged (label 1 in the UCI file); keep as is.
        data["banknote"] = (X, y.astype(int))
    except Exception as exc:
        print(f"banknote skipped: {exc}")

    return data


def make_models(seed):
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=10, random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=seed
        ),
        "PILOT": PILOTClassifier(
            max_depth=4, min_samples_leaf=10, random_state=seed
        ),
        "RaFFLE": RaFFLEClassifier(
            n_estimators=30, max_depth=4, min_samples_leaf=10, random_state=seed
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="one seed only")
    args = parser.parse_args()
    seeds = SEEDS[:1] if args.quick else SEEDS

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    class_counts = []

    for seed in seeds:
        for ds_name, (X, y) in load_datasets(seed).items():
            y = np.asarray(y)

            # 60/20/20 stratified split.
            X_rest, X_test, y_rest, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=seed
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_rest, y_rest, test_size=0.25, stratify=y_rest,
                random_state=seed
            )

            class_counts.append({
                "dataset": ds_name,
                "seed": seed,
                "n": len(y),
                "positive": int(y.sum()),
                "negative": int(len(y) - y.sum()),
            })

            for model_name, model in make_models(seed).items():
                start = time.perf_counter()

                # Training only; validation is not used for choosing settings.
                model.fit(X_train, y_train)

                fit_seconds = time.perf_counter() - start

                # Count discarded logistic candidates for the RaFFLE forest.
                nonconverged_candidates = 0

                if model_name == "RaFFLE":
                    nonconverged_candidates = sum(
                        getattr(estimator, "_nonconverged_candidates", 0)
                        for estimator in model.estimators_
                    )

                    print(
                        f"{ds_name:16s} seed={seed} "
                        f"RaFFLE discarded candidates: "
                        f"{nonconverged_candidates}",
                        flush=True,
                    )

                proba = model.predict_proba(X_test)
                positive_col = list(model.classes_).index(1)
                p_pos = proba[:, positive_col]
                pred = model.predict(X_test)

                rows.append({
                    "dataset": ds_name,
                    "seed": seed,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro"),
                    "roc_auc": roc_auc_score(y_test, p_pos),
                    "log_loss": log_loss(
                        y_test,
                        np.column_stack([1 - p_pos, p_pos]),
                        labels=[0, 1],
                    ),
                    "fit_seconds": fit_seconds,
                    "nonconverged_candidates": nonconverged_candidates,
                })

                print(
                    f"{ds_name:16s} seed={seed} {model_name:12s} "
                    f"acc={rows[-1]['accuracy']:.3f} "
                    f"auc={rows[-1]['roc_auc']:.3f} "
                    f"ll={rows[-1]['log_loss']:.3f} "
                    f"t={fit_seconds:.2f}s",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "per_seed_results.csv", index=False)

    pd.DataFrame(class_counts).to_csv(
        OUT / "class_counts.csv", index=False
    )

    summary = results.groupby(["dataset", "model"])[
        ["accuracy", "macro_f1", "roc_auc", "log_loss", "fit_seconds"]
    ].agg(["mean", "std"]).round(3)

    summary.to_csv(OUT / "summary.csv")

    print("\nSummary written to", OUT / "summary.csv")
    print("Per-seed results written to", OUT / "per_seed_results.csv")


if __name__ == "__main__":
    main()
