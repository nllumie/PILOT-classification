"""Local, feature-level explanations for the Step 5 classifiers.

Three explanation methods, all producing the same ``Explanation`` structure:

* ``explain_logreg``        exact, logit space:   contribution_j = coef_j * z_j
* ``explain_random_forest`` exact, probability space (treeinterpreter method):
                            each split along a tree's path changes the leaf
                            value; that change is credited to the split feature,
                            then averaged over trees.
* ``explain_raffle``        exact per tree, logit space: every node correction on
                            the path is credited to the node's feature (constant
                            parts go to ``baseline``), then averaged over trees.
                            The forest itself averages *probabilities*, so
                            ``sigmoid(baseline + sum(contributions))`` is not
                            identical to the forest probability. The difference
                            is stored in ``approximation_gap`` and reported.

Attribution rules for a PILOT node (see pilot_classification/tree.py):
  con    : coefficient[0]                      -> baseline
  lin    : coefficient[0]                      -> baseline
           coefficient[1] * z                  -> feature
  pcon   : coefficient[side]                   -> feature
  plogit : coefficient[side] + coefficient[side+1] * z -> feature
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import expit, logit
from sklearn.pipeline import Pipeline


@dataclass
class Explanation:
    model_name: str
    predicted_class: object
    probability_positive: float
    space: str                      # "logit" or "probability"
    exact: bool
    baseline: float
    contributions: Dict[int, float] # feature index -> contribution
    approximation_gap: float = 0.0  # 0 for exact methods
    note: str = ""
    per_tree: Optional[List[Dict[int, float]]] = field(default=None, repr=False)

    def total(self) -> float:
        return self.baseline + float(sum(self.contributions.values()))

    def main_feature(self) -> Optional[int]:
        """Feature with the largest absolute contribution, or None if all zero."""
        if not self.contributions:
            return None
        idx = max(self.contributions, key=lambda j: abs(self.contributions[j]))
        return idx if self.contributions[idx] != 0.0 else None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _split_pipeline(model):
    """Return (preprocessing-or-None, final estimator)."""
    if isinstance(model, Pipeline):
        return model[:-1] if len(model.steps) > 1 else None, model.steps[-1][1]
    return None, model


def _prepare_row(model, x: Sequence[float]) -> np.ndarray:
    pre, _ = _split_pipeline(model)
    row = np.asarray(x, dtype=float).reshape(1, -1)
    return pre.transform(row) if pre is not None else row


def _positive_column(estimator) -> int:
    classes = list(estimator.classes_)
    if len(classes) != 2:
        raise ValueError("Explanations are defined for binary classifiers only.")
    return 1  # classes_ is sorted; the second entry is the positive class


# --------------------------------------------------------------------------
# logistic regression
# --------------------------------------------------------------------------

def explain_logreg(model, x: Sequence[float], name: str = "LogReg") -> Explanation:
    pre, lr = _split_pipeline(model)
    z = _prepare_row(model, x)[0]
    coef = np.asarray(lr.coef_).ravel()
    intercept = float(np.asarray(lr.intercept_).ravel()[0])
    contributions = {j: float(coef[j] * z[j]) for j in range(z.size)}
    score = intercept + sum(contributions.values())
    p = float(expit(score))
    pred = lr.classes_[1] if p > 0.5 else lr.classes_[0]
    return Explanation(name, pred, p, "logit", True, intercept, contributions,
                       note="coef_j * standardized_value_j; exact")


# --------------------------------------------------------------------------
# random forest (treeinterpreter decomposition, probability space)
# --------------------------------------------------------------------------

def _sklearn_tree_contributions(tree, z: np.ndarray, pos: int):
    t = tree.tree_
    values = t.value[:, 0, :]
    values = values / values.sum(axis=1, keepdims=True)  # class fractions per node
    node = 0
    baseline = float(values[0, pos])
    contrib: Dict[int, float] = {}
    while t.children_left[node] != -1:
        f = int(t.feature[node])
        child = (t.children_left[node] if z[f] <= t.threshold[node]
                 else t.children_right[node])
        delta = float(values[child, pos] - values[node, pos])
        contrib[f] = contrib.get(f, 0.0) + delta
        node = child
    return baseline, contrib, float(values[node, pos])


def explain_random_forest(model, x: Sequence[float],
                          name: str = "RandomForest") -> Explanation:
    _, rf = _split_pipeline(model)
    z = _prepare_row(model, x)[0]
    pos = _positive_column(rf)
    n = len(rf.estimators_)
    baseline = 0.0
    contributions: Dict[int, float] = {}
    per_tree = []
    for est in rf.estimators_:
        b, c, _ = _sklearn_tree_contributions(est, z, pos)
        baseline += b / n
        for j, v in c.items():
            contributions[j] = contributions.get(j, 0.0) + v / n
        per_tree.append(c)
    p = baseline + sum(contributions.values())
    pred = rf.classes_[1] if p > 0.5 else rf.classes_[0]
    return Explanation(name, pred, float(p), "probability", True, baseline,
                       contributions, per_tree=per_tree,
                       note="path-based change in leaf class fraction, averaged over trees; exact")


# --------------------------------------------------------------------------
# RaFFLE (PILOT classification forest)
# --------------------------------------------------------------------------

def explain_pilot_tree(tree, z_local: np.ndarray):
    """Exact logit decomposition of one PILOTClassifier on one row.

    Returns (baseline_logit, {local_feature: contribution}, final_logit).
    """
    baseline = float(tree._initial_score)
    contrib: Dict[int, float] = {}
    node = tree.tree_
    while node.kind != "terminal":
        c = node.coefficients
        if node.kind == "con":
            baseline += float(c[0])
            break
        if node.kind == "lin":
            zj = (z_local[node.feature] - node.mean) / node.scale
            baseline += float(c[0])
            contrib[node.feature] = contrib.get(node.feature, 0.0) + float(c[1] * zj)
            node = node.left
            continue
        is_left = z_local[node.feature] <= node.threshold
        if node.kind == "pcon":
            term = float(c[0 if is_left else 1])
        else:  # plogit
            off = 0 if is_left else 2
            zj = (z_local[node.feature] - node.mean) / node.scale
            term = float(c[off] + c[off + 1] * zj)
        contrib[node.feature] = contrib.get(node.feature, 0.0) + term
        node = node.left if is_left else node.right
    return baseline, contrib, baseline + sum(contrib.values())


def explain_raffle(model, x: Sequence[float], name: str = "RaFFLE") -> Explanation:
    _, forest = _split_pipeline(model)
    z = _prepare_row(model, x)[0]
    n = len(forest.estimators_)
    baseline = 0.0
    contributions: Dict[int, float] = {}
    per_tree = []
    tree_logits = []
    for est, feats in zip(forest.estimators_, forest.estimators_features_):
        b, c, s = explain_pilot_tree(est, z[feats])
        baseline += b / n
        mapped = {int(feats[j]): v for j, v in c.items()}
        for j, v in mapped.items():
            contributions[j] = contributions.get(j, 0.0) + v / n
        per_tree.append(mapped)
        tree_logits.append(s)
    # forest probability = mean of tree probabilities (what predict_proba does)
    p_forest = float(np.mean(expit(np.asarray(tree_logits))))
    p_from_mean_logit = float(expit(baseline + sum(contributions.values())))
    pred = forest.classes_[1] if p_forest > 0.5 else forest.classes_[0]
    return Explanation(name, pred, p_forest, "logit", False, baseline,
                       contributions, approximation_gap=abs(p_forest - p_from_mean_logit),
                       per_tree=per_tree,
                       note=("exact per tree; forest-level = average logit contribution. "
                             "Forest probability averages tree probabilities, so "
                             "sigmoid(mean logit) differs slightly (see approximation_gap)."))


# --------------------------------------------------------------------------
# faithfulness checks
# --------------------------------------------------------------------------

def check_faithfulness(model, x: Sequence[float], expl: Explanation, atol: float = 1e-8):
    """Compare an explanation with the model's own predict_proba/predict.

    Returns a dict; 'ok' is True when the model's predicted class matches and
    the reconstructed probability matches within tolerance (exact methods) or
    within the recorded approximation gap (RaFFLE).
    """
    row = np.asarray(x, dtype=float).reshape(1, -1)
    p_model = float(model.predict_proba(row)[0, 1])
    pred_model = model.predict(row)[0]
    if expl.space == "logit":
        p_recon = float(expit(expl.total()))
    else:
        p_recon = expl.total()
    gap = abs(p_model - p_recon)
    tol = atol if expl.exact else expl.approximation_gap + atol
    return {
        "model": expl.model_name,
        "prob_model": p_model,
        "prob_explanation": expl.probability_positive,
        "prob_reconstructed": p_recon,
        "gap": gap,
        "class_match": pred_model == expl.predicted_class,
        "ok": (pred_model == expl.predicted_class) and gap <= tol
             and abs(p_model - expl.probability_positive) <= atol,
    }


# --------------------------------------------------------------------------
# text card
# --------------------------------------------------------------------------

def format_card(expl: Explanation, x: Sequence[float], feature_names: Sequence[str],
                class_names: Dict[object, str], top_k: int = 4) -> str:
    """Plain-language explanation card with identical layout for every model."""
    pos_name = class_names[sorted(class_names)[1]]
    neg_name = class_names[sorted(class_names)[0]]
    items = sorted(expl.contributions.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    lines = ["Inputs: " + "; ".join(f"{feature_names[j]} = {float(v):.2f}"
                                    for j, v in enumerate(x)),
             f"Model prediction: {class_names[expl.predicted_class]} "
             f"(P({pos_name}) = {expl.probability_positive:.2f})",
             "Explanation:"]
    for j, v in items:
        direction = pos_name if v > 0 else neg_name
        lines.append(f"  - {feature_names[j]} pushes the prediction toward "
                     f"{direction} (strength {abs(v):.2f})")
    if not items:
        lines.append("  - no feature changed the prediction from the baseline")
    lines.append(f"Combined result favours {class_names[expl.predicted_class]}.")
    if not expl.exact:
        lines.append("(approximate forest-level attribution)")
    return "\n".join(lines)
