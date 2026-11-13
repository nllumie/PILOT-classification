import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from experiments.classification import run_balance_conformal
from experiments.classification.run_conformal import MODEL_NAMES, make_four_way_split


class BalanceConformalTests(unittest.TestCase):
    def test_generator_is_deterministic_and_has_requested_class_counts(self):
        first = run_balance_conformal.generate_synthetic_datasets()
        second = run_balance_conformal.generate_synthetic_datasets()

        self.assertEqual(set(first), set(run_balance_conformal.CONDITIONS))
        for name, (features, labels) in first.items():
            other_features, other_labels = second[name]
            np.testing.assert_array_equal(features, other_features)
            np.testing.assert_array_equal(labels, other_labels)
            self.assertEqual(
                np.bincount(labels, minlength=2).tolist(),
                (np.asarray(run_balance_conformal.CONDITIONS[name]) * 5000)
                .astype(int)
                .tolist(),
            )

    def test_imbalanced_four_way_split_preserves_minority_in_each_partition(self):
        labels = np.repeat([0, 1], [4500, 500])

        partitions, stratified = make_four_way_split(labels, seed=42)

        self.assertTrue(all(stratified.values()))
        all_indices = np.concatenate(list(partitions.values()))
        self.assertEqual(np.unique(all_indices).size, labels.size)
        self.assertEqual(np.sort(all_indices).tolist(), list(range(labels.size)))
        for indices in partitions.values():
            counts = np.bincount(labels[indices], minlength=2)
            self.assertGreater(counts[1], 0)
            self.assertLess(abs(counts[1] / len(indices) - 0.1), 0.01)

    def test_runner_uses_each_condition_and_seed_and_writes_combined_summary(self):
        expected_rows = [
            {"dataset": dataset, "seed": seed, "model": model}
            for dataset in run_balance_conformal.CONDITIONS
            for seed in (42, 52, 62)
            for model in MODEL_NAMES
        ]
        returned_summaries = []

        def fake_evaluate(dataset, features, labels, seed, output_dir):
            summary = pd.DataFrame(
                [row for row in expected_rows if row["dataset"] == dataset and row["seed"] == seed]
            )
            returned_summaries.append((dataset, seed, len(features), len(labels)))
            return pd.DataFrame(), summary

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with mock.patch.object(
                run_balance_conformal,
                "evaluate_dataset",
                side_effect=fake_evaluate,
            ) as evaluate:
                combined = run_balance_conformal.run_experiment(
                    n_samples=100,
                    seeds=(42, 52, 62),
                    output_dir=output_dir,
                )

            saved = pd.read_csv(output_dir / "summary_all_conditions.csv")

        self.assertEqual(evaluate.call_count, 6)
        self.assertEqual(len(returned_summaries), 6)
        self.assertEqual(set(combined["dataset"]), set(run_balance_conformal.CONDITIONS))
        self.assertEqual(set(combined["seed"]), {42, 52, 62})
        self.assertEqual(len(combined), 18)
        pd.testing.assert_frame_equal(combined, saved)


if __name__ == "__main__":
    unittest.main()
