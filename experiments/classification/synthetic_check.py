import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pilot_classification import PILOTClassifier, RaFFLEClassifier


def datasets(seed):
    X_lin, y_lin = make_classification(
        n_samples=600, n_features=6, n_informative=3, n_redundant=0,
        flip_y=0.05, class_sep=1.2, random_state=seed,
    )
    X_nl, y_nl = make_moons(n_samples=600, noise=0.25, random_state=seed)
    return {"linear": (X_lin, y_lin), "moons": (X_nl, y_nl)}


def models(seed):
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "PILOT": PILOTClassifier(max_depth=4, min_samples_leaf=10, random_state=seed),
        "RaFFLE": RaFFLEClassifier(n_estimators=30, max_depth=4, min_samples_leaf=10, random_state=seed),
    }


for name, (X, y) in datasets(42).items():
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"\n{name}")
    for model_name, model in models(42).items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = model.predict(X_te)
        print(f"  {model_name:12s} acc={accuracy_score(y_te, pred):.3f} "
              f"f1={f1_score(y_te, pred, average='macro'):.3f} "
              f"auc={roc_auc_score(y_te, proba):.3f} "
              f"logloss={log_loss(y_te, proba):.3f}")