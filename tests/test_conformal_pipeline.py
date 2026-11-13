import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from experiments.classification import run_conformal


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fit_labels = []
    fit_features = []
    probability_features = []

    def fit(self, X, y):
        type(self).fit_labels.append(np.asarray(y).copy())
        type(self).fit_features.append(np.asarray(X).copy())
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        type(self).probability_features.append(np.asarray(X).copy())
        positive = np.full(X.shape[0], 0.7)
        return np.column_stack((1 - positive, positive))

    def predict(self, X):
        return np.full(X.shape[0], self.classes_[1])


class ConformalPipelineTests(unittest.TestCase):
    def test_split_is_disjoint_complete_and_near_requested_proportions(self):
        y = np.array([0] * 100 + [1] * 100)

        partitions, _ = run_conformal.make_four_way_split(y, seed=42)
        all_indices = np.concatenate(list(partitions.values()))

        self.assertEqual(len(all_indices), len(y))
        self.assertEqual(len(np.unique(all_indices)), len(y))
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(y)))
        self.assertEqual(
            {name: len(indices) for name, indices in partitions.items()},
            {"train": 100, "validation": 30, "calibration": 30, "test": 40},
        )

    def test_split_stratifies_partitions(self):
        y = np.array([0] * 100 + [1] * 100)

        partitions, stratified = run_conformal.make_four_way_split(y, seed=52)

        self.assertTrue(all(stratified.values()))
        for indices in partitions.values():
            self.assertEqual(np.bincount(y[indices]).tolist()[0], len(indices) // 2)
            self.assertEqual(np.bincount(y[indices]).tolist()[1], len(indices) // 2)

    def test_split_can_fall_back_when_stratification_is_not_feasible(self):
        y = np.array([0, 0, 0, 1])

        partitions, stratified = run_conformal.make_four_way_split(y, seed=42)

        self.assertEqual(sum(map(len, partitions.values())), len(y))
        self.assertFalse(all(stratified.values()))

    def test_saved_indices_are_original_partition_indices(self):
        y = np.array([0] * 20 + [1] * 20)
        partitions, _ = run_conformal.make_four_way_split(y, seed=42)

        with tempfile.TemporaryDirectory() as directory:
            path = run_conformal._save_split_indices(
                "toy", 42, partitions, Path(directory)
            )
            saved = pd.read_csv(path)

        self.assertEqual(set(saved["partition"]), set(run_conformal.PARTITION_NAMES))
        for partition_name, indices in partitions.items():
            actual = saved.loc[
                saved["partition"] == partition_name, "original_index"
            ].to_numpy()
            np.testing.assert_array_equal(np.sort(actual), np.sort(indices))

    def test_only_training_rows_and_labels_are_used_to_fit_models(self):
        RecordingClassifier.fit_labels = []
        RecordingClassifier.fit_features = []
        RecordingClassifier.probability_features = []
        y = np.array([0] * 50 + [1] * 50)
        X = np.column_stack((np.arange(y.size, dtype=float), np.ones(y.size)))
        X[0, 1] = np.nan
        partitions, _ = run_conformal.make_four_way_split(y, seed=42)
        expected_train_labels = y[partitions["train"]]

        def models(_seed):
            return {
                model_name: Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RecordingClassifier()),
                ])
                for model_name in run_conformal.MODEL_NAMES
            }

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(run_conformal, "make_models", side_effect=models):
                run_conformal.evaluate_dataset(
                    "toy", X, y, 42, output_dir=Path(directory)
                )

        self.assertEqual(len(RecordingClassifier.fit_labels), 3)
        for fitted_labels in RecordingClassifier.fit_labels:
            np.testing.assert_array_equal(fitted_labels, expected_train_labels)
        expected_train_features = X[partitions["train"], 0]
        for fitted_features in RecordingClassifier.fit_features:
            np.testing.assert_array_equal(
                fitted_features[:, 0], expected_train_features
            )
        expected_calibration_features = X[partitions["calibration"], 0]
        expected_test_features = X[partitions["test"], 0]
        for call_index, probability_features in enumerate(
            RecordingClassifier.probability_features
        ):
            expected_features = (
                expected_calibration_features
                if call_index % 3 == 0
                else expected_test_features
            )
            np.testing.assert_array_equal(
                probability_features[:, 0], expected_features
            )

    def test_same_split_indices_are_saved_for_all_model_results(self):
        y = np.array([0] * 30 + [1] * 30)
        X = np.column_stack((np.arange(y.size, dtype=float), np.arange(y.size) % 3))

        class TinyClassifier(RecordingClassifier):
            fit_labels = []
            fit_features = []
            probability_features = []

        def models(_seed):
            return {
                model_name: Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", TinyClassifier()),
                ])
                for model_name in run_conformal.MODEL_NAMES
            }

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(run_conformal, "make_models", side_effect=models):
                predictions, summary = run_conformal.evaluate_dataset(
                    "toy", X, y, 52, output_dir=Path(directory)
                )
            split_indices = pd.read_csv(
                Path(directory) / "split_indices_toy_seed52.csv"
            )

        expected_test = set(
            split_indices.loc[
                split_indices["partition"] == "test", "original_index"
            ]
        )
        for _, group in predictions.groupby("model"):
            self.assertEqual(set(group["original_index"]), expected_test)
        self.assertEqual(len(summary), len(run_conformal.MODEL_NAMES))
        self.assertTrue((summary["test_count_class_0"] > 0).all())
        self.assertTrue((summary["test_count_class_1"] > 0).all())
        self.assertTrue(summary["overall_coverage"].between(0.0, 1.0).all())
        self.assertTrue((summary["alpha"] == run_conformal.ALPHA).all())


if __name__ == "__main__":
    unittest.main()
