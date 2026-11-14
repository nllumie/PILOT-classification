"""Write one booklet per participant. Run: python -m step7_interpretability.make_booklets"""
import pathlib
import pandas as pd

HERE = pathlib.Path(__file__).parent
OUT = HERE / "booklets"; OUT.mkdir(exist_ok=True)
cases = pd.read_csv(HERE / "explanation_cases.csv").set_index("case_id")
orders = pd.read_csv(HERE / "order_assignments.csv")
intro = (HERE / "participant_questions.txt").read_text(encoding="utf-8").split("PRACTICE")[0]
names = {"LogReg": "Model A", "RandomForest": "Model B", "RaFFLE": "Model C"}  # blind labels
PAGE = "\n" + "=" * 60 + "  (next page)\n\n"

for pid, rows in orders.groupby("participant_id"):
    parts = [f"Participant {pid}\n\n{intro}"]
    for model in rows.sort_values("slot").model.unique():
        prac = cases[(cases.model == model) & (cases.practice)].iloc[0]
        parts.append(f"PRACTICE for {names[model]} (not scored)\n\n{prac.card_text}")
    for _, r in rows.sort_values("slot").iterrows():
        c = cases.loc[r.case_id]
        m = names[r.model]
        parts.append(f"Case {r.slot} of 6 ({m})\n{c.inputs_text}\n\n"
                     f"Q1  What will {m} predict for this note: genuine or forged?  ______")
        parts.append(f"Case {r.slot} of 6 ({m}): explanation\n{c.card_text}\n\n"
                     "Q2  Which ONE measurement contributes most strongly to THIS prediction?  ______\n"
                     "Q3  In one or two sentences, why did this model choose that class?\n\n\n"
                     "Q4  How easy was this explanation to understand? 1 2 3 4 5")
    parts.append("Which model's explanations helped you most, and what was confusing?\n\n\n")
    (OUT / f"{pid}.txt").write_text(PAGE.join(parts), encoding="utf-8")
print("wrote", len(list(OUT.glob("*.txt"))), "booklets to", OUT)