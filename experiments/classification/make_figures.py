# Figures for the report: confusion matrices, ROC curves, and a
# RaFFLE "training curve" (accuracy and log loss vs number of trees).
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from experiments.classification.run_classification import (
    OUT, load_datasets, make_models,
)

DATASET = sys.argv[1] if len(sys.argv) > 1 else "banknote"
SEED = 42

X, y = load_datasets(42, selected=[DATASET])[DATASET]
X, y = np.asarray(X, float), np.asarray(y, int)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Fit the same four models the report compares (same settings as the runner).
models = make_models(SEED)
fitted = {name: models[name].fit(X_tr, y_tr)
          for name in ["LogReg", "RandomForest", "PILOT", "RaFFLE"]}

# 1. Confusion matrices (rows = true, columns = predicted; 1 = positive).
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (name, model) in zip(axes, fitted.items()):
    cm = confusion_matrix(y_te, model.predict(X_te), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title(name)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.suptitle(f"Confusion matrices: {DATASET} (seed {SEED}, 80/20 split)")
fig.tight_layout()
fig.savefig(OUT / f"{DATASET}_confusion.png", dpi=300)

# 2. ROC curves on the test split.
fig, ax = plt.subplots(figsize=(5, 5))
for name, model in fitted.items():
    p = model.predict_proba(X_te)[:, list(model.classes_).index(1)]
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title(f"ROC curves: {DATASET}")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / f"{DATASET}_roc.png", dpi=300)

# 3. RaFFLE "training curve": average the first k trees, no refitting.
raffle = fitted["RaFFLE"]
Xt = raffle.named_steps["imputer"].transform(X_te)
forest = raffle.named_steps["model"]
per_tree = [est.predict_proba(Xt[:, feats])
            for est, feats in zip(forest.estimators_, forest.estimators_features_)]
ks = np.arange(1, len(per_tree) + 1)
acc = [np.mean(forest.classes_[np.mean(per_tree[:k], axis=0).argmax(1)] == y_te)
       for k in ks]
ll = [log_loss(y_te, np.mean(per_tree[:k], axis=0), labels=[0, 1]) for k in ks]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(ks, acc, marker="o", markersize=3)
ax1.set_xlabel("Number of trees"); ax1.set_ylabel("Test accuracy")
ax2.plot(ks, ll, marker="o", markersize=3, color="tab:orange")
ax2.set_xlabel("Number of trees"); ax2.set_ylabel("Test log loss")
fig.suptitle(f"RaFFLE accuracy and log loss vs. number of trees: {DATASET}")
fig.tight_layout()
fig.savefig(OUT / f"{DATASET}_raffle_tree_curve.png", dpi=300)

print("Saved figures for", DATASET, "in", OUT)