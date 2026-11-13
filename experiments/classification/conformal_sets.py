"""Split conformal prediction sets for classifiers with ``predict_proba``."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np


class SplitConformalClassifier:
    """Calibrate and predict class sets using a frozen probabilistic model.

    Calibration and prediction use the class-column order in the model's
    ``classes_`` attribute. The model is never fitted or otherwise modified.
    """

    def __init__(self, model: Any, alpha: float = 0.10) -> None:
        self.model = model
        self.alpha = alpha

    @staticmethod
    def _validate_alpha(alpha: object) -> float:
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a real number strictly between 0 and 1")
        value = float(alpha)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("alpha must be finite and strictly between 0 and 1")
        return value

    @staticmethod
    def _validate_X(X: object, name: str) -> np.ndarray:
        try:
            values = np.asarray(X)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a two-dimensional array") from exc
        if values.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if values.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one observation")
        if values.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one feature")
        return values

    @staticmethod
    def _validate_classes(model: Any) -> np.ndarray:
        try:
            classes = np.asarray(model.classes_)
        except AttributeError as exc:
            raise ValueError("model must be fitted and expose classes_") from exc
        if classes.ndim != 1 or classes.size < 2:
            raise ValueError(
                "model.classes_ must be one-dimensional with at least two classes"
            )
        for index, label in enumerate(classes):
            if any(SplitConformalClassifier._labels_equal(label, earlier)
                   for earlier in classes[:index]):
                raise ValueError("model.classes_ must contain unique labels")
        return classes.copy()

    @staticmethod
    def _labels_equal(left: Any, right: Any) -> bool:
        try:
            equal = left == right
            return bool(equal) if np.ndim(equal) == 0 else False
        except (TypeError, ValueError):
            return False

    @classmethod
    def _class_index(cls, label: Any, classes: np.ndarray) -> int:
        for index, candidate in enumerate(classes):
            if cls._labels_equal(candidate, label):
                return index
        raise ValueError(f"calibration label {label!r} is not in model.classes_")

    @staticmethod
    def _validate_y(y: object, n_samples: int) -> np.ndarray:
        try:
            labels = np.asarray(y)
        except (TypeError, ValueError) as exc:
            raise TypeError("y_cal must be a one-dimensional array of labels") from exc
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        elif labels.ndim != 1:
            raise ValueError(
                "y_cal must be one-dimensional or a single-column array"
            )
        if labels.shape[0] != n_samples:
            raise ValueError("X_cal and y_cal must have matching sample counts")
        if labels.size == 0:
            raise ValueError("y_cal must contain at least one label")
        return labels

    @staticmethod
    def _predict_probabilities(
        model: Any,
        X: np.ndarray,
        classes: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.shape != (X.shape[0], classes.size):
            raise ValueError(
                "predict_proba must return one row per observation and one column per class"
            )
        try:
            probabilities = np.asarray(probabilities, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("predict_proba must return numerical probabilities") from exc
        if not np.isfinite(probabilities).all():
            raise ValueError("predict_proba must return only finite probabilities")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError("predict_proba values must be between 0 and 1")
        if not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8
        ):
            raise ValueError("predict_proba rows must sum to 1")
        return probabilities

    def calibrate(self, X_cal: object, y_cal: object) -> "SplitConformalClassifier":
        """Calibrate the wrapper on held-out examples without fitting the model."""
        alpha = self._validate_alpha(self.alpha)
        X_values = self._validate_X(X_cal, "X_cal")
        labels = self._validate_y(y_cal, X_values.shape[0])
        classes = self._validate_classes(self.model)
        probabilities = self._predict_probabilities(self.model, X_values, classes)
        class_indices = np.fromiter(
            (self._class_index(label, classes) for label in labels),
            dtype=np.intp,
            count=labels.size,
        )
        scores = 1.0 - probabilities[np.arange(labels.size), class_indices]
        k = math.ceil((labels.size + 1) * (1.0 - alpha))
        sorted_scores = np.sort(scores)
        threshold = float(sorted_scores[k - 1]) if k <= labels.size else math.inf

        self.alpha_ = alpha
        self.classes_ = classes
        self.n_calibration_ = int(labels.size)
        self.n_features_in_ = int(X_values.shape[1])
        self.k_ = k
        self.calibration_scores_ = scores.copy()
        self.threshold_ = threshold
        return self

    def predict_sets(self, X_new: object) -> list[list[Any]]:
        """Return each observation's prediction set in ``classes_`` order."""
        if not hasattr(self, "threshold_"):
            raise ValueError("calibrate must be called before predict_sets")
        X_values = self._validate_X(X_new, "X_new")
        if X_values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_new has {X_values.shape[1]} features; calibration expects "
                f"{self.n_features_in_}"
            )
        probabilities = self._predict_probabilities(
            self.model, X_values, self.classes_
        )
        return [
            [
                self.classes_[class_index]
                for class_index, probability in enumerate(row)
                if 1.0 - probability <= self.threshold_
            ]
            for row in probabilities
        ]
