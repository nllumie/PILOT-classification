"""Bootstrap ensemble of additive-logistic PILOT classification trees."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .tree import PILOTClassifier


class RaFFLEClassifier(ClassifierMixin, BaseEstimator):
    """Random forest of binary ``PILOTClassifier`` trees.

    Rows are sampled with replacement for each tree. Optional tree-level
    feature sampling is followed by each tree's node-level feature sampling.
    Forest probabilities are the mean of tree probabilities.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        random_state: Optional[int] = None,
        max_features_tree: Optional[int] = None,
        max_features_node: Optional[int] = None,
        lambda_0: float = 1.0,
        lambda_1: float = 1.0,
        alpha: float = 0.5,
        max_depth: int = 5,
        max_model_depth: int = 10,
        min_samples_fit: int = 10,
        min_samples_leaf: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features_tree = max_features_tree
        self.max_features_node = max_features_node
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.alpha = alpha
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.min_samples_fit = min_samples_fit
        self.min_samples_leaf = min_samples_leaf

    @staticmethod
    def _validate_count(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return int(value)

    def _validate_parameters(self) -> None:
        self._validate_count("n_estimators", self.n_estimators)
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, (int, np.integer))
        ):
            raise TypeError("random_state must be an integer or None")
        for name in ("max_features_tree", "max_features_node"):
            value = getattr(self, name)
            if value is not None:
                self._validate_count(name, value)

        # Reuse the tree's public parameter validation rather than duplicating it.
        PILOTClassifier(
            lambda_0=self.lambda_0,
            lambda_1=self.lambda_1,
            alpha=self.alpha,
            max_depth=self.max_depth,
            max_model_depth=self.max_model_depth,
            min_samples_fit=self.min_samples_fit,
            min_samples_leaf=self.min_samples_leaf,
        )._validate_parameters()

    def fit(self, X: object, y: object) -> "RaFFLEClassifier":
        """Fit trees to bootstrap samples while retaining global label order."""
        self._validate_parameters()
        values = PILOTClassifier._validate_X(X)
        labels, classes = PILOTClassifier._validate_y(y, values.shape[0])
        n_samples, n_features = values.shape
        if (
            self.max_features_tree is not None
            and self.max_features_tree > n_features
        ):
            raise ValueError(
                "max_features_tree cannot exceed the number of input features"
            )
        if (
            self.max_features_node is not None
            and self.max_features_node > n_features
        ):
            raise ValueError(
                "max_features_node cannot exceed the number of input features"
            )

        rng = np.random.default_rng(self.random_state)
        self.classes_ = classes.copy()
        self.n_features_in_ = n_features
        self.estimators_ = []
        self.estimators_features_ = []

        for _ in range(self.n_estimators):
            bootstrap_indices = rng.integers(0, n_samples, size=n_samples)
            if self.max_features_tree is None:
                feature_indices = np.arange(n_features, dtype=np.int64)
            else:
                feature_indices = np.sort(
                    rng.choice(
                        n_features, size=self.max_features_tree, replace=False
                    )
                )

            node_feature_count = self.max_features_node
            if node_feature_count is not None:
                node_feature_count = min(node_feature_count, feature_indices.size)
            estimator = PILOTClassifier(
                lambda_0=self.lambda_0,
                lambda_1=self.lambda_1,
                alpha=self.alpha,
                max_depth=self.max_depth,
                max_model_depth=self.max_model_depth,
                min_samples_fit=self.min_samples_fit,
                min_samples_leaf=self.min_samples_leaf,
                max_features_node=node_feature_count,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            estimator._fit_with_global_classes(
                values[np.ix_(bootstrap_indices, feature_indices)],
                labels[bootstrap_indices],
                self.classes_,
            )
            estimator.feature_indices_ = feature_indices.copy()
            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_indices.copy())

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "estimators_") or not self.estimators_:
            raise ValueError("This RaFFLEClassifier instance is not fitted yet")

    def predict_proba(self, X: object) -> np.ndarray:
        """Average tree probabilities, keeping columns in ``classes_`` order."""
        self._check_is_fitted()
        values = PILOTClassifier._validate_X(
            X, expected_features=self.n_features_in_
        )
        predictions = [
            estimator.predict_proba(values[:, feature_indices])
            for estimator, feature_indices in zip(
                self.estimators_, self.estimators_features_
            )
        ]
        return np.mean(predictions, axis=0)

    def predict(self, X: object) -> np.ndarray:
        """Return the most probable class for each observation."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
