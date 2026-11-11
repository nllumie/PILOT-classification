import unittest
from unittest import mock

import numpy as np

from pilot_classification import PILOTClassifier, RaFFLEClassifier


def make_dataset():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(36, 3))
    y = np.where(1.7 * X[:, 0] - 0.4 * X[:, 1] > 0, "yes", "no")
    return X, y


class PILOTClassifierTests(unittest.TestCase):
    def make_classifier(self, **overrides):
        parameters = {
            "lambda_0": 0.2,
            "lambda_1": 0.2,
            "alpha": 0.25,
            "max_depth": 2,
            "max_model_depth": 4,
            "min_samples_fit": 6,
            "min_samples_leaf": 4,
            "random_state": 7,
        }
        parameters.update(overrides)
        return PILOTClassifier(**parameters)

    def test_fit_produces_valid_probabilities_and_labels(self):
        X, y = make_dataset()
        classifier = self.make_classifier().fit(X, y)

        probabilities = classifier.predict_proba(X)
        predictions = classifier.predict(X)

        self.assertEqual(probabilities.shape, (X.shape[0], 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(classifier.classes_, np.array(["no", "yes"]))
        self.assertEqual(classifier.n_features_in_, X.shape[1])
        self.assertEqual(classifier._n_features_in, classifier.n_features_in_)
        np.testing.assert_array_equal(
            predictions, classifier.classes_[np.argmax(probabilities, axis=1)]
        )

    def test_invalid_input_dimensions_and_values_are_rejected(self):
        X, y = make_dataset()
        classifier = self.make_classifier()

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            classifier.fit(X[:, 0], y)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            classifier.fit(X, y.reshape(-1, 2))
        with self.assertRaisesRegex(ValueError, "finite"):
            classifier.fit(np.where(np.indices(X.shape)[0] == 0, np.nan, X), y)
        with self.assertRaisesRegex(ValueError, "mismatched sample counts"):
            classifier.fit(X, y[:-1])
        with self.assertRaisesRegex(ValueError, "exactly two"):
            classifier.fit(X, np.full(y.shape, "no"))
        with self.assertRaisesRegex(TypeError, "numerical"):
            classifier.fit(np.full(X.shape, "feature"), y)

    def test_invalid_regularization_is_rejected(self):
        for parameter in ("lambda_0", "lambda_1"):
            for value in (0.0, -1.0, np.inf, np.nan):
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaisesRegex(ValueError, "strictly positive"):
                        self.make_classifier(**{parameter: value})
            with self.subTest(parameter=parameter, value="invalid"):
                with self.assertRaisesRegex(TypeError, "finite positive"):
                    self.make_classifier(**{parameter: "invalid"})

    def test_one_class_child_uses_only_regularized_constant_correction(self):
        X = np.array([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        classifier = self.make_classifier(
            lambda_0=0.01,
            lambda_1=0.01,
            alpha=0.0,
            max_depth=1,
            max_model_depth=4,
            min_samples_fit=2,
            min_samples_leaf=2,
        ).fit(X, y)

        self.assertIn(classifier.tree_.kind, {"pcon", "plogit"})

        def check_one_class_branches(node, row_indices):
            if node is None or node.kind == "terminal":
                return
            branch_labels = y[row_indices]
            if np.all(branch_labels == branch_labels[0]):
                self.assertEqual(node.kind, "con")
                return
            if node.kind == "con":
                return
            if node.kind == "lin":
                check_one_class_branches(node.left, row_indices)
                return
            left = X[row_indices, node.feature] <= node.threshold
            check_one_class_branches(node.left, row_indices[left])
            check_one_class_branches(node.right, row_indices[~left])

        root_left = X[:, classifier.tree_.feature] <= classifier.tree_.threshold
        check_one_class_branches(
            classifier.tree_.left, np.flatnonzero(root_left)
        )
        check_one_class_branches(
            classifier.tree_.right, np.flatnonzero(~root_left)
        )
        probabilities = classifier.predict_proba(X)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities > 0.0) & (probabilities < 1.0)).all())

    def test_undersized_one_class_node_terminates_without_correction(self):
        X = np.arange(6, dtype=np.float64).reshape(-1, 1)
        y = np.array([0, 1, 0, 1, 0, 1])
        classifier = self.make_classifier(
            max_model_depth=0, min_samples_fit=3
        ).fit(X, y)

        classifier.max_model_depth = 2
        classifier._binary_y = np.array([0.0, 0.0])
        classifier._training_scores = np.array([3.0, 3.0])
        scores_before = classifier._training_scores.copy()
        node = classifier._grow(
            np.array([0, 1], dtype=np.int64),
            split_depth=0,
            model_depth=0,
        )

        self.assertEqual(node.kind, "terminal")
        np.testing.assert_array_equal(classifier._training_scores, scores_before)

    def test_depth_and_sample_size_stopping_rules(self):
        X, y = make_dataset()
        no_splits = self.make_classifier(max_depth=0).fit(X, y)

        def contains_split(node):
            if node is None or node.kind == "terminal":
                return False
            return node.kind in {"pcon", "plogit"} or contains_split(
                node.left
            ) or contains_split(node.right)

        self.assertFalse(contains_split(no_splits.tree_))

        too_small = self.make_classifier(min_samples_fit=X.shape[0] + 1).fit(X, y)
        self.assertEqual(too_small.tree_.kind, "terminal")
        self.assertEqual(too_small.predict_proba(X[:2]).shape, (2, 2))

        no_corrections = self.make_classifier(max_model_depth=0).fit(X, y)
        self.assertEqual(no_corrections.tree_.kind, "terminal")

    def test_zero_variance_features_are_not_fitted_as_slopes_or_splits(self):
        X = np.ones((12, 1))
        y = np.array([0, 1] * 6)
        classifier = self.make_classifier(
            max_depth=3, min_samples_fit=4
        ).fit(X, y)

        self.assertIn(classifier.tree_.kind, {"con", "terminal"})
        self.assertTrue(np.isfinite(classifier.predict_proba(X)).all())

        X_with_constant = np.column_stack(
            (np.ones(20), np.linspace(-2.0, 2.0, 20))
        )
        y_separated = np.array([0] * 10 + [1] * 10)
        linear_only = self.make_classifier(
            lambda_0=0.1,
            lambda_1=0.1,
            alpha=0.0,
            max_depth=0,
            max_model_depth=2,
            min_samples_fit=4,
            min_samples_leaf=2,
            max_features_node=1,
        ).fit(X_with_constant, y_separated)
        self.assertEqual(linear_only.tree_.kind, "lin")
        self.assertEqual(linear_only.tree_.feature, 1)

    def test_piecewise_logistic_correction_and_prediction_traversal(self):
        rng = np.random.default_rng(0)
        feature = np.linspace(-3.0, 3.0, 80)
        logits = np.where(
            feature <= 0.0,
            -0.5 + 2.8 * feature,
            0.3 + 2.8 * feature,
        )
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))
        X = feature.reshape(-1, 1)
        classifier = PILOTClassifier(
            lambda_0=0.001,
            lambda_1=0.001,
            alpha=0.0,
            max_depth=1,
            max_model_depth=3,
            min_samples_fit=10,
            min_samples_leaf=8,
            random_state=0,
        ).fit(X, y)

        root = classifier.tree_
        self.assertEqual(root.kind, "plogit")
        self.assertEqual(root.left.kind, "terminal")
        self.assertEqual(root.right.kind, "terminal")

        standardized = (feature - root.mean) / root.scale
        is_left = feature <= root.threshold
        expected_scores = np.full(feature.shape, classifier._initial_score)
        expected_scores[is_left] += (
            root.coefficients[0] + root.coefficients[1] * standardized[is_left]
        )
        expected_scores[~is_left] += (
            root.coefficients[2] + root.coefficients[3] * standardized[~is_left]
        )
        expected_positive_probability = 1.0 / (1.0 + np.exp(-expected_scores))

        probabilities = classifier.predict_proba(X)
        np.testing.assert_allclose(
            probabilities[:, 1], expected_positive_probability
        )
        np.testing.assert_allclose(
            probabilities[:, 0], 1.0 - expected_positive_probability
        )

    def test_same_seed_reproduces_tree_and_predictions(self):
        X, y = make_dataset()
        first = self.make_classifier().fit(X, y)
        second = self.make_classifier().fit(X, y)

        np.testing.assert_array_equal(
            first.predict_proba(X), second.predict_proba(X)
        )

    def test_prediction_accepts_new_rows_with_matching_feature_count(self):
        X, y = make_dataset()
        classifier = self.make_classifier().fit(X, y)
        new_rows = np.array([[0.1, -0.3, 1.2], [-1.1, 0.4, 0.0]])

        self.assertEqual(classifier.predict_proba(new_rows).shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "expects 3"):
            classifier.predict_proba(new_rows[:, :2])

    def test_private_fit_with_one_class_preserves_global_probability_order(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([1, 1])
        classes = np.array([0, 1])
        classifier = self.make_classifier(
            lambda_0=0.001,
            alpha=0.0,
            max_depth=1,
            max_model_depth=3,
            min_samples_fit=2,
            min_samples_leaf=1,
        )

        classifier._fit_with_global_classes(X, y, classes)

        np.testing.assert_array_equal(classifier.classes_, classes)
        self.assertEqual(classifier.tree_.kind, "con")
        probabilities = classifier.predict_proba(X)
        self.assertEqual(probabilities.shape, (X.shape[0], 2))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

        correction = classifier.tree_.coefficients[0]
        expected_logit = classifier._initial_score + correction
        expected_positive_probability = 1.0 / (1.0 + np.exp(-expected_logit))
        np.testing.assert_allclose(probabilities[:, 0], 1.0 - expected_positive_probability)
        np.testing.assert_allclose(probabilities[:, 1], expected_positive_probability)
        self.assertGreater(correction, 0.0)
        regularized_gradient = (
            2.0 * X.shape[0] * (expected_positive_probability - 1.0)
            + 2.0 * classifier.lambda_0 * correction
        )
        self.assertAlmostEqual(regularized_gradient, 0.0, places=6)

        with self.assertRaisesRegex(ValueError, "exactly two"):
            self.make_classifier().fit(X, y)

    def test_parameters_are_revalidated_at_fit(self):
        X, y = make_dataset()
        classifier = self.make_classifier()
        classifier.lambda_0 = 0.0

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            classifier.fit(X, y)


class RaFFLEClassifierTests(unittest.TestCase):
    def make_forest(self, **overrides):
        parameters = {
            "n_estimators": 5,
            "random_state": 11,
            "max_features_tree": 2,
            "max_features_node": 1,
            "lambda_0": 0.2,
            "lambda_1": 0.2,
            "alpha": 0.25,
            "max_depth": 2,
            "max_model_depth": 4,
            "min_samples_fit": 5,
            "min_samples_leaf": 3,
        }
        parameters.update(overrides)
        return RaFFLEClassifier(**parameters)

    def test_forest_probabilities_and_predictions(self):
        X, y = make_dataset()
        forest = self.make_forest().fit(X, y)

        probabilities = forest.predict_proba(X)
        predictions = forest.predict(X)

        self.assertEqual(probabilities.shape, (X.shape[0], 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(forest.classes_, np.array(["no", "yes"]))
        np.testing.assert_array_equal(
            predictions, forest.classes_[np.argmax(probabilities, axis=1)]
        )
        self.assertEqual(len(forest.estimators_), forest.n_estimators)

    def test_forest_reproducibility(self):
        X, y = make_dataset()
        first = self.make_forest().fit(X, y)
        second = self.make_forest().fit(X, y)

        np.testing.assert_array_equal(
            first.predict_proba(X), second.predict_proba(X)
        )
        for first_features, second_features in zip(
            first.estimators_features_, second.estimators_features_
        ):
            np.testing.assert_array_equal(first_features, second_features)

    def test_one_class_bootstrap_keeps_global_probability_columns(self):
        rng = np.random.default_rng(23)
        X = rng.normal(size=(20, 2))
        y = np.array(["negative"] * 19 + ["positive"])
        forest = RaFFLEClassifier(
            n_estimators=30,
            random_state=4,
            max_depth=1,
            max_model_depth=2,
            min_samples_fit=3,
            min_samples_leaf=2,
        ).fit(X, y)

        one_class_estimators = [
            estimator
            for estimator in forest.estimators_
            if np.unique(estimator._binary_y).size == 1
        ]
        self.assertTrue(one_class_estimators)
        for estimator in one_class_estimators:
            np.testing.assert_array_equal(
                estimator.classes_, forest.classes_
            )
            probabilities = estimator.predict_proba(X[:3])
            self.assertEqual(probabilities.shape, (3, 2))
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

        forest_probabilities = forest.predict_proba(X[:3])
        self.assertEqual(forest_probabilities.shape, (3, 2))
        np.testing.assert_allclose(forest_probabilities.sum(axis=1), 1.0)

    def test_feature_subsampling_uses_valid_reproducible_subsets(self):
        X, y = make_dataset()
        forest = self.make_forest(max_features_tree=2).fit(X, y)

        self.assertEqual(len(forest.estimators_features_), forest.n_estimators)
        for feature_indices in forest.estimators_features_:
            self.assertEqual(feature_indices.size, 2)
            self.assertEqual(np.unique(feature_indices).size, 2)
            self.assertTrue(np.all(feature_indices >= 0))
            self.assertTrue(np.all(feature_indices < X.shape[1]))
        for estimator, feature_indices in zip(
            forest.estimators_, forest.estimators_features_
        ):
            self.assertEqual(estimator.n_features_in_, feature_indices.size)
            self.assertLessEqual(
                estimator.max_features_node, feature_indices.size
            )

    def test_predict_proba_is_mean_of_tree_probabilities(self):
        X, y = make_dataset()
        forest = self.make_forest(n_estimators=7).fit(X, y)

        tree_probabilities = [
            estimator.predict_proba(X[:, feature_indices])
            for estimator, feature_indices in zip(
                forest.estimators_, forest.estimators_features_
            )
        ]
        expected = np.mean(tree_probabilities, axis=0)
        averaged_hard_votes = np.mean(
            [
                np.eye(2)[np.argmax(probabilities, axis=1)]
                for probabilities in tree_probabilities
            ],
            axis=0,
        )

        np.testing.assert_allclose(forest.predict_proba(X), expected)
        self.assertFalse(np.allclose(expected, averaged_hard_votes))

    def test_node_feature_sampling_is_independent_and_reproducible(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(120, 4))
        y = ((X[:, 0] + 0.8 * X[:, 1] + 0.3 * X[:, 2]) > 0).astype(int)
        parameters = {
            "lambda_0": 0.01,
            "lambda_1": 0.01,
            "alpha": 0.0,
            "max_depth": 2,
            "max_model_depth": 5,
            "min_samples_fit": 8,
            "min_samples_leaf": 4,
            "max_features_node": 2,
            "random_state": 8,
        }

        original_best_candidate = PILOTClassifier._best_candidate

        def fit_and_record():
            records = []

            def recording_best_candidate(estimator, indices, split_depth):
                generator = estimator._rng

                class RecordingRNG:
                    def choice(self, available, size=None, replace=True):
                        selected = generator.choice(
                            available, size=size, replace=replace
                        )
                        records.append(
                            (
                                indices.copy(),
                                np.asarray(available).copy(),
                                np.asarray(selected).copy(),
                            )
                        )
                        return selected

                estimator._rng = RecordingRNG()
                try:
                    return original_best_candidate(estimator, indices, split_depth)
                finally:
                    estimator._rng = generator

            with mock.patch.object(
                PILOTClassifier,
                "_best_candidate",
                recording_best_candidate,
            ):
                estimator = PILOTClassifier(**parameters).fit(X, y)
            return estimator, records

        first, first_records = fit_and_record()
        second, second_records = fit_and_record()

        self.assertGreaterEqual(len(first_records), 2)
        self.assertEqual(len(first_records), len(second_records))
        for first_record, second_record in zip(first_records, second_records):
            first_indices, available, selected = first_record
            second_indices, second_available, second_selected = second_record
            self.assertTrue(np.all(available >= 0))
            self.assertTrue(np.all(available < X.shape[1]))
            self.assertGreater(selected.size, 0)
            self.assertLessEqual(selected.size, parameters["max_features_node"])
            self.assertEqual(np.unique(selected).size, selected.size)
            self.assertTrue(np.isin(selected, available).all())
            np.testing.assert_array_equal(first_indices, second_indices)
            np.testing.assert_array_equal(available, second_available)
            np.testing.assert_array_equal(selected, second_selected)
        self.assertGreater(
            len({tuple(record[2].tolist()) for record in first_records}), 1
        )
        self.assertGreater(
            len({tuple(record[0].tolist()) for record in first_records}), 1
        )
        np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))

    def test_invalid_forest_inputs_and_prediction_shape(self):
        X, y = make_dataset()
        forest = self.make_forest()

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            forest.fit(X[:, 0], y)
        with self.assertRaisesRegex(ValueError, "finite"):
            bad_X = X.copy()
            bad_X[0, 0] = np.inf
            forest.fit(bad_X, y)
        with self.assertRaisesRegex(ValueError, "mismatched sample counts"):
            forest.fit(X, y[:-1])
        with self.assertRaisesRegex(ValueError, "exactly two"):
            forest.fit(X, np.zeros(y.size))
        with self.assertRaisesRegex(ValueError, "expects 3"):
            forest.fit(X, y).predict_proba(X[:, :2])
        with self.assertRaisesRegex(ValueError, "max_features_tree"):
            self.make_forest(max_features_tree=X.shape[1] + 1).fit(X, y)

    def test_forest_predicts_new_rows(self):
        X, y = make_dataset()
        forest = self.make_forest().fit(X, y)
        new_rows = np.array([[0.1, -0.3, 1.2], [-1.1, 0.4, 0.0]])

        self.assertEqual(forest.predict_proba(new_rows).shape, (2, 2))
        self.assertEqual(forest.predict(new_rows).shape, (2,))


if __name__ == "__main__":
    unittest.main()
