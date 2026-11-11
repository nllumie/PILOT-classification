import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

from pilot_classification import PILOTClassifier, RaFFLEClassifier

SEED = 42
OUT = pathlib.Path("experiments/classification/results")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "spambase_results.csv"

data = fetch_ucirepo(id=94)
X = data.data.features.to_numpy(dtype=float)
y = data.data.targets.iloc[:, 0].to_numpy().astype(int)   # 1 = spam (positive)
print("rows", X.shape, "spam fraction", round(float(y.mean()), 3), flush=True)

# Same 60/20/20 protocol as run_classification.py; validation is unused here.
X_rest, X_test, y_rest, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
X_train, _, y_train, _ = train_test_split(
    X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=SEED)

models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=SEED),
    "PILOT": PILOTClassifier(max_depth=4, min_samples_leaf=10, max_features_node=7, random_state=SEED),
    "RaFFLE": RaFFLEClassifier(n_estimators=30, max_depth=4, min_samples_leaf=10,
                               max_features_node=7, random_state=SEED),
}

rows = []
for name, model in models.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    p_pos = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    pred = model.predict(X_test)
    discarded = sum(getattr(e, "_nonconverged_candidates", 0)
                    for e in getattr(model, "estimators_", []))

    rows.append({
        "dataset": "spambase", "seed": SEED, "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, p_pos),
        "log_loss": log_loss(y_test, np.column_stack([1 - p_pos, p_pos]), labels=[0, 1]),
        "fit_seconds": fit_seconds,
        "nonconverged_candidates": discarded,
    })
    # Save after every model so partial progress survives an interruption.
    pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"{name:12s} acc={rows[-1]['accuracy']:.3f} auc={rows[-1]['roc_auc']:.3f} "
          f"ll={rows[-1]['log_loss']:.3f} t={fit_seconds:.1f}s discarded={discarded}", flush=True)

print("Done. Results in", RESULTS, flush=True)