"""Compare split conformal classification under balanced and imbalanced priors."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from experiments.classification.run_classification import SEEDS
from experiments.classification.run_conformal import evaluate_dataset


N_SAMPLES = 5000
DATA_SEED = 42
OUTPUT_DIR = pathlib.Path("experiments/classification/outputs/synthetic_balance")
CONDITIONS = {
    "synthetic_balanced": [0.5, 0.5],
    "synthetic_imbalanced": [0.9, 0.1],
}


def generate_synthetic_datasets(
    n_samples: int = N_SAMPLES,
    random_state: int = DATA_SEED,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate the same classification process with two class priors."""
    datasets = {}
    for name, weights in CONDITIONS.items():
        features, labels = make_classification(
            n_samples=n_samples,
            n_features=6,
            n_informative=3,
            n_redundant=0,
            class_sep=1.2,
            flip_y=0.0,
            weights=weights,
            random_state=random_state,
        )
        datasets[name] = (features, labels)
    return datasets


def run_experiment(
    n_samples: int = N_SAMPLES,
    data_seed: int = DATA_SEED,
    seeds: tuple[int, ...] | list[int] = tuple(SEEDS),
    output_dir: pathlib.Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Evaluate each class-prior condition for every configured split seed."""
    if not seeds:
        raise ValueError("at least one split seed is required")

    summaries = []
    for dataset_name, (features, labels) in generate_synthetic_datasets(
        n_samples=n_samples,
        random_state=data_seed,
    ).items():
        counts = np.bincount(labels, minlength=2)
        expected_counts = np.asarray(CONDITIONS[dataset_name]) * n_samples
        if not np.array_equal(counts, expected_counts.astype(int)):
            raise RuntimeError(
                f"{dataset_name} generated class counts {counts.tolist()}, "
                f"expected {expected_counts.astype(int).tolist()}"
            )

        for seed in seeds:
            _, summary = evaluate_dataset(
                dataset_name,
                features,
                labels,
                seed,
                output_dir=output_dir,
            )
            summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "summary_all_conditions.csv", index=False)
    print("Saved:", output_dir / "summary_all_conditions.csv")
    return combined


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
