"""Build the Step 7 study materials from the Step 5 models (Banknote dataset).

Run from the PILOT root:  python -m step7_interpretability.build_cases

Writes into step7_interpretability/:
  models.joblib            fitted LogReg / RandomForest / RaFFLE (seed 42, 80/20 split)
  explanation_cases.csv    what participants see (inputs text, card text, order slot)
  answer_key.csv           model prediction, main feature, true label (PRIVATE)
  faithfulness_report.csv  card-vs-model check for every case
  order_assignments.csv    model/case order for participants P01..P24
"""
from __future__ import annotations

import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.classification.explanations import (
    check_faithfulness, explain_logreg, explain_raffle,
    explain_random_forest, format_card,
)
from experiments.classification.run_classification import load_datasets, make_models

HERE = pathlib.Path(__file__).parent
SEED = 42
FEATURES = ["variance", "skewness", "curtosis", "entropy"]
FEATURE_DESCRIPTIONS = {
    "variance": "how much the image texture varies",
    "skewness": "how lopsided the texture distribution is",
    "curtosis": "how peaked / heavy-tailed the texture distribution is",
    "entropy": "how random the texture is",
}
CLASS_NAMES = {0: "genuine", 1: "forged"}      # 1 = positive class (Step 5 choice)
EXPLAINERS = {"LogReg": explain_logreg, "RandomForest": explain_random_forest,
              "RaFFLE": explain_raffle}
PREFIX = {"LogReg": "LR", "RandomForest": "RF", "RaFFLE": "RA"}
MAX_GAP = 0.05   # RaFFLE cards shown to participants must have approximation_gap <= this
MODEL_ORDERS = [["RaFFLE", "RandomForest", "LogReg"],
                ["RandomForest", "LogReg", "RaFFLE"],
                ["LogReg", "RaFFLE", "RandomForest"]]


def fit_models(reuse=False):
    path = HERE / "models.joblib"
    if reuse and path.exists():
        saved = joblib.load(path)
        print("loaded fitted models from", path, flush=True)
        return saved["models"], saved["X_test"], saved["y_test"]
    X, y = load_datasets(SEED, selected=["banknote"])["banknote"]
    X, y = np.asarray(X, float), np.asarray(y, int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=SEED)
    models = make_models(SEED)
    fitted = {}
    for name in EXPLAINERS:
        print(f"fitting {name} ...", flush=True)
        fitted[name] = models[name].fit(X_tr, y_tr)
    joblib.dump({"models": fitted, "X_test": X_te, "y_test": y_te}, HERE / "models.joblib")
    return fitted, X_te, y_te


def candidate_table(fitted, X_te, y_te):
    rows = []
    for name, model in fitted.items():
        for i, x in enumerate(X_te):
            e = EXPLAINERS[name](model, x)
            vals = sorted((abs(v) for v in e.contributions.values()), reverse=True)
            margin = vals[0] / vals[1] if len(vals) > 1 and vals[1] > 0 else np.inf
            rows.append({"model": name, "test_index": i, "pred": int(e.predicted_class),
                         "p_pos": e.probability_positive, "main": e.main_feature(),
                         "margin": margin, "true": int(y_te[i]), "gap": e.approximation_gap})
    return pd.DataFrame(rows)


def pick_cases(cands, n_practice=1):
    """Two study cases + one practice case per model, no test row reused.

    Clear and practice cases must have approximation_gap <= MAX_GAP. A 'close'
    case (confidence 0.60-0.85) is where RaFFLE's trees disagree, so its gap can
    exceed MAX_GAP; the smallest-gap close candidate is used and the gap is
    printed and stored in answer_key.csv. Main-feature margin >= 1.5 always.
    """
    used = set()
    chosen = []
    for model in EXPLAINERS:
        c_all = cands[(cands.model == model) & (cands.margin >= 1.5)].copy()
        c_all["conf"] = np.maximum(c_all.p_pos, 1 - c_all.p_pos)
        c_all = c_all.sort_values(["gap", "test_index"])
        c_ok = c_all[c_all.gap <= MAX_GAP]
        clear = c_ok[c_ok.conf >= 0.9]
        close = c_all[(c_all.conf > 0.6) & (c_all.conf < 0.85)]
        picks = []
        for pool, want_pred, kind in [(clear, 1, "clear"), (close, 0, "close")]:
            p = pool[(pool.pred == want_pred) & ~pool.test_index.isin(used)]
            if p.empty:
                p = pool[~pool.test_index.isin(used)]
            if p.empty:
                raise RuntimeError(f"no {kind} case for {model}")
            r = p.iloc[0]
            if kind == "close" and r.gap > MAX_GAP:
                print(f"NOTE {model}: close case {int(r.test_index)} has gap "
                      f"{r.gap:.3f} > {MAX_GAP}; smallest available", flush=True)
            used.add(int(r.test_index)); picks.append((r, kind, False))
        prac = c_ok[~c_ok.test_index.isin(used)].iloc[0]
        used.add(int(prac.test_index)); picks.append((prac, "practice", True))
        chosen.extend([(model,) + t for t in picks])
    return chosen

def main():
    import sys
    fitted, X_te, y_te = fit_models(reuse="--reuse" in sys.argv)
    cands = candidate_table(fitted, X_te, y_te)
    chosen = pick_cases(cands)

    cases, key, faith = [], [], []
    for model, r, kind, practice in chosen:
        i = int(r.test_index); x = X_te[i]
        e = EXPLAINERS[model](fitted[model], x)
        card = format_card(e, x, FEATURES, CLASS_NAMES)
        inputs_text = card.split("\n")[0]
        case_id = f"{PREFIX[model]}{'P' if practice else ''}{i}"
        cases.append({"case_id": case_id, "model": model, "practice": practice,
                      "kind": kind, **{f: float(v) for f, v in zip(FEATURES, x)},
                      "inputs_text": inputs_text, "card_text": card})
        main_j = e.main_feature()
        key.append({"case_id": case_id, "model": model, "practice": practice,
                    "model_prediction": CLASS_NAMES[int(e.predicted_class)],
                    "prob_forged": round(e.probability_positive, 4),
                    "main_feature": FEATURES[main_j],
                    "main_direction": CLASS_NAMES[1] if e.contributions[main_j] > 0 else CLASS_NAMES[0],
                    "contributions": {FEATURES[j]: round(v, 4) for j, v in e.contributions.items()},
                    "true_label": CLASS_NAMES[int(y_te[i])],
                    "approximation_gap": round(e.approximation_gap, 6)})
        f = check_faithfulness(fitted[model], x, e); f["case_id"] = case_id; faith.append(f)

    pd.DataFrame(cases).to_csv(HERE / "explanation_cases.csv", index=False)
    pd.DataFrame(key).to_csv(HERE / "answer_key.csv", index=False)
    fdf = pd.DataFrame(faith); fdf.to_csv(HERE / "faithfulness_report.csv", index=False)

    # participant order assignments: cycle model orders, alternate case order
    rows = []
    for p in range(24):
        order = MODEL_ORDERS[p % 3]
        flip = (p // 3) % 2 == 1
        slot = 0
        for m in order:
            ids = [c["case_id"] for c in cases if c["model"] == m and not c["practice"]]
            if flip: ids = ids[::-1]
            for cid in ids:
                slot += 1
                rows.append({"participant_id": f"P{p+1:02d}", "slot": slot, "model": m, "case_id": cid})
    pd.DataFrame(rows).to_csv(HERE / "order_assignments.csv", index=False)

    print("\nFaithfulness:")
    print(fdf[["case_id", "model", "prob_model", "prob_explanation", "gap", "ok"]].to_string(index=False))
    print("\nAll cards faithful:", bool(fdf.ok.all()))
    print("Max RaFFLE approximation gap among shown cards:",
          round(max(k["approximation_gap"] for k in key if k["model"] == "RaFFLE"), 4),
          "(limit", MAX_GAP, ")")
    print("\nAnswer key (private):")
    print(pd.DataFrame(key)[["case_id", "model", "practice", "model_prediction", "prob_forged",
                             "main_feature", "main_direction", "true_label"]].to_string(index=False))
    print("\nExample card:\n" + cases[0]["card_text"])


if __name__ == "__main__":
    main()
