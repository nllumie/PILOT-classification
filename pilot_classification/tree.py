"""Binary PILOT-style classification tree with additive logistic corrections."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


_BASE_COMPLEXITY = {
    "con": 1.0,
    "lin": 2.0,
    "pcon": 5.0,
    "plogit": 7.0,
}


@dataclass
class _Node:
    kind: str
    feature: Optional[int] = None
    threshold: Optional[float] = None
    mean: Optional[float] = None
    scale: Optional[float] = None
    coefficients: Optional[np.ndarray] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


@dataclass
class _Candidate:
    kind: str
    delta: float
    correction: np.ndarray
    coefficients: np.ndarray
    feature: Optional[int] = None
    threshold: Optional[float] = None
    mean: Optional[float] = None
    scale: Optional[float] = None


class PILOTClassifier:
    """Greedy binary classification tree using additive logistic corrections.

    The implementation supports numerical features only. Its local complexity
    penalties are experimental heuristics and do not inherit regression PILOT
    or RaFFLE theoretical guarantees.
    """

    def __init__(
        self,
        lambda_0: float = 1.0,
        lambda_1: float = 1.0,
        alpha: float = 0.5,
        max_depth: int = 5,
        max_model_depth: int = 10,
        min_samples_fit: int = 10,
        min_samples_leaf: int = 5,
        max_features_node: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.alpha = alpha
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.min_samples_fit = min_samples_fit
        self.min_samples_leaf = min_samples_leaf
        self.max_features_node = max_features_node
        self.random_state = random_state
        self._validate_parameters()

    @staticmethod
    def _positive_finite_parameter(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite positive number")
        result = float(value)
        if not np.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
        return result

    @staticmethod
    def _nonnegative_integer(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return int(value)

    @staticmethod
    def _positive_integer(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return int(value)

    def _validate_parameters(self) -> None:
        self._positive_finite_parameter("lambda_0", self.lambda_0)
        self._positive_finite_parameter("lambda_1", self.lambda_1)

        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("alpha must be a finite number in [0, 1]")
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1]")

        self._nonnegative_integer("max_depth", self.max_depth)
        self._nonnegative_integer("max_model_depth", self.max_model_depth)
        self._positive_integer("min_samples_fit", self.min_samples_fit)
        self._positive_integer("min_samples_leaf", self.min_samples_leaf)

        if self.max_features_node is not None:
            self._positive_integer("max_features_node", self.max_features_node)
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, (int, np.integer))
        ):
            raise TypeError("random_state must be an integer or None")

    @staticmethod
    def _validate_X(X: object, expected_features: Optional[int] = None) -> np.ndarray:
        try:
            values = np.asarray(X)
        except (TypeError, ValueError) as exc:
            raise TypeError("X must be a two-dimensional numerical matrix") from exc

        if values.ndim != 2:
            raise ValueError(f"X must be two-dimensional; received {values.ndim} dimensions")
        if values.shape[0] == 0:
            raise ValueError("X must contain at least one observation")
        if values.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        if np.iscomplexobj(values):
            raise TypeError("X must contain real-valued numerical features")
        if values.dtype.kind in "US":
            raise TypeError("X must contain only numerical features")
        try:
            values = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("X must contain only numerical features") from exc
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite feature values")
        if expected_features is not None and values.shape[1] != expected_features:
            raise ValueError(
                f"X has {values.shape[1]} features; the fitted model expects "
                f"{expected_features}"
            )
        return values

    @staticmethod
    def _validate_y(y: object, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            labels = np.asarray(y)
        except (TypeError, ValueError) as exc:
            raise TypeError("y must be a one-dimensional array of class labels") from exc

        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        elif labels.ndim != 1:
            raise ValueError("y must be one-dimensional or a single-column array")
        if labels.shape[0] != n_samples:
            raise ValueError(
                f"X and y have mismatched sample counts ({n_samples} and {labels.shape[0]})"
            )
        if labels.size == 0:
            raise ValueError("y must contain at least one class label")
        if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
            raise ValueError("y must not contain NaN or infinite labels")

        try:
            classes = np.unique(labels)
        except (TypeError, ValueError) as exc:
            raise TypeError("y labels must be mutually comparable") from exc
        if classes.size != 2:
            raise ValueError(
                f"y must contain exactly two global classes; found {classes.size}"
            )
        return labels, classes

    @staticmethod
    def _standardize(values: np.ndarray) -> tuple[float, float, Optional[np.ndarray]]:
        """Return a stable node-local mean, scale, and standardized values."""
        magnitude = float(np.max(np.abs(values)))
        if magnitude == 0.0:
            return 0.0, 0.0, None

        mean = float(magnitude * np.mean(values / magnitude))
        deviations = values - mean
        deviation_scale = float(np.max(np.abs(deviations)))
        if deviation_scale == 0.0:
            return mean, 0.0, None

        scale = float(
            deviation_scale
            * np.sqrt(np.mean(np.square(deviations / deviation_scale)))
        )
        if not np.isfinite(scale) or scale <= 0.0:
            return mean, 0.0, None
        return mean, scale, deviations / scale

    @staticmethod
    def _twice_negative_log_likelihood(
        scores: np.ndarray, labels: np.ndarray
    ) -> float:
        return float(
            2.0 * np.sum(np.logaddexp(0.0, scores) - labels * scores)
        )

    def _fit_coefficients(
        self,
        design: np.ndarray,
        current_scores: np.ndarray,
        labels: np.ndarray,
        penalties: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fit one correction and fail explicitly if optimization is unsuccessful."""

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            scores = current_scores + design @ coefficients
            loss = self._twice_negative_log_likelihood(scores, labels)
            loss += float(np.dot(penalties, np.square(coefficients)))
            gradient = (
                2.0 * design.T @ (expit(scores) - labels)
                + 2.0 * penalties * coefficients
            )
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(
                "Logistic correction optimization did not converge: "
                f"{result.message}"
            )
        coefficients = np.asarray(result.x, dtype=np.float64)
        if not np.isfinite(coefficients).all() or not np.isfinite(result.fun):
            raise RuntimeError(
                "Logistic correction optimization returned non-finite coefficients "
                "or objective"
            )
        return coefficients, float(result.fun)

    def _complexity_penalty(self, kind: str, n_samples: int) -> float:
        degrees_of_freedom = 1.0 + self.alpha * (_BASE_COMPLEXITY[kind] - 1.0)
        return float(degrees_of_freedom * np.log(n_samples))

    def _make_candidate(
        self,
        kind: str,
        design: np.ndarray,
        current_scores: np.ndarray,
        labels: np.ndarray,
        penalties: np.ndarray,
        baseline_loss: float,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        mean: Optional[float] = None,
        scale: Optional[float] = None,
    ) -> _Candidate:
        coefficients, penalized_objective = self._fit_coefficients(
            design, current_scores, labels, penalties
        )
        delta = (
            penalized_objective
            - baseline_loss
            + self._complexity_penalty(kind, labels.size)
        )
        return _Candidate(
            kind=kind,
            delta=delta,
            correction=design @ coefficients,
            coefficients=coefficients,
            feature=feature,
            threshold=threshold,
            mean=mean,
            scale=scale,
        )

    def _best_candidate(
        self, indices: np.ndarray, split_depth: int
    ) -> Optional[_Candidate]:
        scores = self._training_scores[indices]
        labels = self._binary_y[indices]
        n_samples = indices.size
        baseline_loss = self._twice_negative_log_likelihood(scores, labels)

        best: Optional[_Candidate] = None

        # A constant correction is always available, including at max_depth.
        intercept_design = np.ones((n_samples, 1), dtype=np.float64)
        best = self._make_candidate(
            "con",
            intercept_design,
            scores,
            labels,
            np.array([self.lambda_0]),
            baseline_loss,
        )
        if np.all(labels == labels[0]):
            return best if best.delta < 0.0 else None

        usable_features = np.flatnonzero(
            np.any(
                self._X[indices] != self._X[indices][0:1, :],
                axis=0,
            )
        )
        if usable_features.size == 0:
            return best if best.delta < 0.0 else None

        feature_limit = self.max_features_node
        if feature_limit is None:
            feature_limit = self._X.shape[1]
        feature_limit = min(feature_limit, usable_features.size)
        considered = self._rng.choice(
            usable_features, size=feature_limit, replace=False
        )

        can_split = split_depth < self.max_depth
        for feature in considered:
            values = self._X[indices, feature]
            mean, scale, standardized = self._standardize(values)

            if standardized is not None:
                linear_design = np.column_stack(
                    (np.ones(n_samples, dtype=np.float64), standardized)
                )
                candidate = self._make_candidate(
                    "lin",
                    linear_design,
                    scores,
                    labels,
                    np.array([self.lambda_0, self.lambda_1]),
                    baseline_loss,
                    feature=int(feature),
                    mean=mean,
                    scale=scale,
                )
                if candidate.delta < (best.delta if best is not None else 0.0):
                    best = candidate

            if not can_split:
                continue

            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            boundaries = np.flatnonzero(sorted_values[:-1] < sorted_values[1:]) + 1
            for left_count in boundaries:
                right_count = n_samples - int(left_count)
                if (
                    left_count < self.min_samples_leaf
                    or right_count < self.min_samples_leaf
                ):
                    continue

                threshold = float(sorted_values[left_count - 1])
                left_mask = values <= threshold
                right_mask = ~left_mask
                pcon_design = np.column_stack((left_mask, right_mask)).astype(
                    np.float64
                )
                candidate = self._make_candidate(
                    "pcon",
                    pcon_design,
                    scores,
                    labels,
                    np.array([self.lambda_0, self.lambda_0]),
                    baseline_loss,
                    feature=int(feature),
                    threshold=threshold,
                )
                if candidate.delta < (best.delta if best is not None else 0.0):
                    best = candidate

                if standardized is not None:
                    plogit_design = np.column_stack(
                        (
                            left_mask,
                            left_mask * standardized,
                            right_mask,
                            right_mask * standardized,
                        )
                    ).astype(np.float64)
                    candidate = self._make_candidate(
                        "plogit",
                        plogit_design,
                        scores,
                        labels,
                        np.array(
                            [
                                self.lambda_0,
                                self.lambda_1,
                                self.lambda_0,
                                self.lambda_1,
                            ]
                        ),
                        baseline_loss,
                        feature=int(feature),
                        threshold=threshold,
                        mean=mean,
                        scale=scale,
                    )
                    if candidate.delta < (
                        best.delta if best is not None else 0.0
                    ):
                        best = candidate

        return best if best is not None and best.delta < 0.0 else None

    def _grow(
        self, indices: np.ndarray, split_depth: int, model_depth: int
    ) -> _Node:
        if model_depth >= self.max_model_depth:
            return _Node("terminal")
        if indices.size < self.min_samples_fit:
            return _Node("terminal")

        labels = self._binary_y[indices]
        is_one_class = np.all(labels == labels[0])

        candidate = self._best_candidate(indices, split_depth)
        if candidate is None:
            return _Node("terminal")

        # Every selected correction is added to the node's current logits.
        self._training_scores[indices] += candidate.correction
        node = _Node(
            kind=candidate.kind,
            feature=candidate.feature,
            threshold=candidate.threshold,
            mean=candidate.mean,
            scale=candidate.scale,
            coefficients=candidate.coefficients,
        )

        if candidate.kind == "con" or is_one_class:
            return node
        if candidate.kind == "lin":
            node.left = self._grow(indices, split_depth, model_depth + 1)
            return node

        values = self._X[indices, candidate.feature]
        left_mask = values <= candidate.threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        next_depth = split_depth + 1
        next_model_depth = model_depth + 1
        node.left = self._grow(left_indices, next_depth, next_model_depth)
        node.right = self._grow(right_indices, next_depth, next_model_depth)
        return node

    def fit(self, X: object, y: object) -> "PILOTClassifier":
        """Fit a binary classifier to numerical features."""
        self._validate_parameters()
        values = self._validate_X(X)
        labels, classes = self._validate_y(y, values.shape[0])
        return self._fit_validated(values, labels, classes)

    def _fit_with_global_classes(
        self, X: object, y: object, classes: np.ndarray
    ) -> "PILOTClassifier":
        """Fit bootstrap data while retaining the forest's global class order."""
        self._validate_parameters()
        values = self._validate_X(X)
        labels = np.asarray(y)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        elif labels.ndim != 1:
            raise ValueError("bootstrap y must be one-dimensional")
        if labels.shape[0] != values.shape[0]:
            raise ValueError("bootstrap X and y must have matching sample counts")

        global_classes = np.asarray(classes)
        if global_classes.ndim != 1 or global_classes.size != 2:
            raise ValueError("global classes must contain exactly two labels")
        if not np.isin(labels, global_classes).all():
            raise ValueError("bootstrap labels must belong to the global classes")

        return self._fit_validated(values, labels, global_classes)

    def _fit_validated(
        self, values: np.ndarray, labels: np.ndarray, classes: np.ndarray
    ) -> "PILOTClassifier":
        if (
            self.max_features_node is not None
            and self.max_features_node > values.shape[1]
        ):
            raise ValueError(
                "max_features_node cannot exceed the number of input features"
            )

        self._X = values
        self._binary_y = (labels == classes[1]).astype(np.float64)
        positive_rate = (float(np.sum(self._binary_y)) + 0.5) / (
            values.shape[0] + 1.0
        )
        self._initial_score = float(np.log(positive_rate / (1.0 - positive_rate)))
        self._training_scores = np.full(
            values.shape[0], self._initial_score, dtype=np.float64
        )
        self._rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = values.shape[1]
        self._n_features_in = self.n_features_in_

        root_indices = np.arange(values.shape[0], dtype=np.int64)
        self.tree_ = self._grow(root_indices, split_depth=0, model_depth=0)
        self.classes_ = classes.copy()
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "tree_") or not hasattr(self, "classes_"):
            raise ValueError("This PILOTClassifier instance is not fitted yet")

    def predict_proba(self, X: object) -> np.ndarray:
        """Return probabilities in the column order given by ``classes_``."""
        self._check_is_fitted()
        values = self._validate_X(X, expected_features=self._n_features_in)
        probabilities = np.empty((values.shape[0], 2), dtype=np.float64)

        for row_index, row in enumerate(values):
            score = self._initial_score
            node = self.tree_
            while node.kind != "terminal":
                coefficients = node.coefficients
                if node.kind == "con":
                    score += float(coefficients[0])
                    break
                if node.kind == "lin":
                    standardized = (row[node.feature] - node.mean) / node.scale
                    score += float(coefficients[0] + coefficients[1] * standardized)
                    node = node.left
                    continue

                is_left = row[node.feature] <= node.threshold
                if node.kind == "pcon":
                    side = 0 if is_left else 1
                    score += float(coefficients[side])
                else:
                    side_offset = 0 if is_left else 2
                    standardized = (row[node.feature] - node.mean) / node.scale
                    score += float(
                        coefficients[side_offset]
                        + coefficients[side_offset + 1] * standardized
                    )
                node = node.left if is_left else node.right

            positive_probability = float(expit(score))
            probabilities[row_index, 0] = 1.0 - positive_probability
            probabilities[row_index, 1] = positive_probability

        return probabilities

    def predict(self, X: object) -> np.ndarray:
        """Predict labels, resolving exact probability ties to ``classes_[0]``."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
