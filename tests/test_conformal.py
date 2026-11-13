import unittest

import numpy as np

from experiments.classification.conformal_sets import SplitConformalClassifier


class DummyClassifier:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities)
        self.fit_calls = 0
        self.predict_proba_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        return self

    def predict_proba(self, X):
        self.predict_proba_calls += 1
        if callable(self.probabilities):
            return self.probabilities(X)
        return self.probabilities


def rows(count):
    return np.arange(count, dtype=float).reshape(-1, 1)


class SplitConformalClassifierTests(unittest.TestCase):
    def test_extracts_true_class_probability_by_classes_order(self):
        model = DummyClassifier(
            ["malignant", "benign"],
            [[0.8, 0.2], [0.1, 0.9]],
        )
        conformal = SplitConformalClassifier(model, alpha=0.34).calibrate(
            rows(2), ["benign", "malignant"]
        )

        np.testing.assert_allclose(conformal.calibration_scores_, [0.8, 0.9])
        self.assertEqual(conformal.k_, 2)
        self.assertEqual(conformal.threshold_, 0.9)
        np.testing.assert_array_equal(conformal.classes_, ["malignant", "benign"])
        self.assertEqual(
            conformal.predict_sets(rows(2)),
            [["malignant", "benign"], ["malignant", "benign"]],
        )

    def test_calculates_finite_sample_quantile(self):
        model = DummyClassifier(
            [0, 1],
            [[0.6, 0.4], [0.9, 0.1], [0.7, 0.3], [0.8, 0.2]],
        )
        conformal = SplitConformalClassifier(model, alpha=0.5).calibrate(
            rows(4), [0, 0, 0, 0]
        )

        self.assertEqual(conformal.n_calibration_, 4)
        self.assertEqual(conformal.k_, 3)
        self.assertAlmostEqual(conformal.threshold_, 0.3)

    def test_uses_infinite_threshold_when_k_exceeds_calibration_size(self):
        model = DummyClassifier([0, 1], [[0.9, 0.1]] * 3)
        conformal = SplitConformalClassifier(model).calibrate(rows(3), [0, 0, 0])

        self.assertEqual(conformal.k_, 4)
        self.assertEqual(conformal.threshold_, float("inf"))
        model.probabilities = np.array([[0.5, 0.5]])
        self.assertEqual(conformal.predict_sets(rows(1)), [[0, 1]])

    def test_predicts_only_labels_with_score_at_or_below_threshold(self):
        model = DummyClassifier(
            [0, 1],
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]],
        )
        conformal = SplitConformalClassifier(model, alpha=0.5).calibrate(
            rows(4), [0, 0, 0, 0]
        )
        model.probabilities = np.array([[0.8, 0.2], [0.2, 0.8]])

        self.assertAlmostEqual(conformal.threshold_, 0.3)
        self.assertEqual(conformal.predict_sets(rows(2)), [[0], [1]])

    def test_allows_empty_prediction_sets(self):
        model = DummyClassifier([0, 1], [[0.9, 0.1], [0.8, 0.2]])
        conformal = SplitConformalClassifier(model, alpha=0.5).calibrate(
            rows(2), [0, 0]
        )
        model.probabilities = np.array([[0.5, 0.5]])

        self.assertAlmostEqual(conformal.threshold_, 0.2)
        self.assertEqual(conformal.predict_sets(rows(1)), [[]])

    def test_prediction_set_can_contain_both_labels(self):
        model = DummyClassifier([0, 1], [[0.2, 0.8]] * 3)
        conformal = SplitConformalClassifier(model, alpha=0.5).calibrate(
            rows(3), [0, 0, 0]
        )
        model.probabilities = np.array([[0.5, 0.5]])

        self.assertEqual(conformal.predict_sets(rows(1)), [[0, 1]])

    def test_prediction_does_not_refit_or_modify_classifier_classes(self):
        model = DummyClassifier(["no", "yes"], [[0.8, 0.2], [0.2, 0.8]])
        original_classes = model.classes_.copy()
        conformal = SplitConformalClassifier(model, alpha=0.5).calibrate(
            rows(2), ["no", "yes"]
        )
        fit_calls_after_calibration = model.fit_calls

        conformal.predict_sets(rows(2))

        self.assertEqual(model.fit_calls, fit_calls_after_calibration)
        self.assertEqual(model.fit_calls, 0)
        np.testing.assert_array_equal(model.classes_, original_classes)

    def test_results_are_deterministic(self):
        model = DummyClassifier(
            ["b", "a"],
            [[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]],
        )
        first = SplitConformalClassifier(model, alpha=0.5).calibrate(
            rows(4), ["a", "b", "b", "a"]
        )
        second = SplitConformalClassifier(model, alpha=0.5).calibrate(
            rows(4), ["a", "b", "b", "a"]
        )

        self.assertEqual(first.threshold_, second.threshold_)
        self.assertEqual(first.predict_sets(rows(4)), second.predict_sets(rows(4)))

    def test_rejects_invalid_alpha(self):
        for alpha in (0, 1, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    SplitConformalClassifier(DummyClassifier([0, 1], []), alpha).calibrate(
                        rows(1), [0]
                    )
        with self.assertRaises(TypeError):
            SplitConformalClassifier(DummyClassifier([0, 1], []), True).calibrate(
                rows(1), [0]
            )

    def test_rejects_invalid_calibration_dimensions_and_labels(self):
        model = DummyClassifier([0, 1], [[0.5, 0.5]] * 2)
        conformal = SplitConformalClassifier(model)

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            conformal.calibrate(np.array([1.0, 2.0]), [0, 1])
        with self.assertRaisesRegex(ValueError, "matching sample counts"):
            conformal.calibrate(rows(2), [0])
        with self.assertRaisesRegex(ValueError, "not in model.classes_"):
            conformal.calibrate(rows(2), [0, 2])

    def test_rejects_invalid_prediction_dimensions_and_unfitted_use(self):
        model = DummyClassifier([0, 1], [[0.5, 0.5]])
        conformal = SplitConformalClassifier(model)
        with self.assertRaisesRegex(ValueError, "calibrate"):
            conformal.predict_sets(rows(1))

        conformal.calibrate(rows(1), [0])
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            conformal.predict_sets(np.array([1.0]))
        with self.assertRaisesRegex(ValueError, "calibration expects"):
            conformal.predict_sets(np.ones((1, 2)))

    def test_rejects_invalid_probability_outputs(self):
        invalid_outputs = [
            np.array([[0.5]]),
            np.array([[np.nan, np.nan]]),
            np.array([[-0.1, 1.1]]),
            np.array([[0.2, 0.2]]),
            np.array([["invalid", "probability"]]),
        ]
        for output in invalid_outputs:
            with self.subTest(output=output):
                model = DummyClassifier([0, 1], output)
                with self.assertRaises((TypeError, ValueError)):
                    SplitConformalClassifier(model, alpha=0.5).calibrate(
                        rows(1), [0]
                    )


if __name__ == "__main__":
    unittest.main()
