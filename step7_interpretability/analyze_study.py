"""Analyse Step 7 responses.  Run from PILOT root: python -m step7_interpretability.analyze_study

Reads responses.csv + answer_key.csv, writes results/participant_level.csv,
results/model_summary.csv and results/by_familiarity.csv.
"""
import pathlib
import pandas as pd

HERE = pathlib.Path(__file__).parent
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)

resp = pd.read_csv(HERE / "responses.csv")
key = pd.read_csv(HERE / "answer_key.csv")
key = key[~key.practice][["case_id", "model", "model_prediction", "main_feature"]]

d = resp.merge(key, on=["case_id", "model"], how="left", validate="m:1")
missing = d.model_prediction.isna()
if missing.any():
    raise ValueError(f"responses reference unknown cases: {d[missing].case_id.unique()}")

d["prediction_correct"] = d.prediction_guess.str.strip().str.lower() == d.model_prediction.str.lower()
d["main_feature_correct"] = d.main_feature_choice.str.strip().str.lower() == d.main_feature.str.lower()

per_participant = (d.groupby(["participant_id", "ml_familiarity", "model"])
                     .agg(n_cases=("case_id", "count"),
                          prediction_correct=("prediction_correct", "mean"),
                          main_feature_correct=("main_feature_correct", "mean"),
                          reason_score=("reason_score_0_2", "mean"),
                          clarity=("clarity_1_to_5", "mean"),
                          prediction_seconds=("prediction_seconds", "median"),
                          explanation_seconds=("explanation_seconds", "median"))
                     .reset_index())
per_participant.to_csv(OUT / "participant_level.csv", index=False)

metrics = ["prediction_correct", "main_feature_correct", "reason_score", "clarity",
           "prediction_seconds", "explanation_seconds"]
summary = per_participant.groupby("model")[metrics].agg(["mean", "std", "count"]).round(3)
summary.to_csv(OUT / "model_summary.csv")
by_fam = per_participant.groupby(["ml_familiarity", "model"])[metrics].agg(["mean", "count"]).round(3)
by_fam.to_csv(OUT / "by_familiarity.csv")

print("Participants:", per_participant.participant_id.nunique(),
      "| responses:", len(d), "| missing answers:", int(d[metrics[:4]].isna().sum().sum()))
print("\nParticipant-level means by model (2 cases each):")
print(summary.to_string())
print("\nBy ML familiarity:")
print(by_fam.to_string())
