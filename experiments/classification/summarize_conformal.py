
import ast
import pathlib
import pandas as pd


OUT = pathlib.Path("experiments/classification/outputs/synthetic_balance")

rows = []

prediction_files = sorted(OUT.glob("predictions_*.csv"))

if not prediction_files:
    raise FileNotFoundError(
        f"No prediction CSV files found in {OUT}"
    )


def parse_conformal_set(value):
    """Convert a CSV conformal_set value into a Python list."""
    if isinstance(value, list):
        return value

    if pd.isna(value):
        raise ValueError("Found a missing conformal_set value.")

    parsed = ast.literal_eval(str(value))

    if not isinstance(parsed, (list, tuple, set)):
        raise ValueError(
            f"Invalid conformal set: {value}"
        )

    return list(parsed)


for file in prediction_files:
    d = pd.read_csv(file)

    required_columns = {
        "dataset",
        "seed",
        "model",
        "true_label",
        "conformal_set",
    }

    missing = required_columns - set(d.columns)

    if missing:
        raise ValueError(
            f"{file.name} is missing columns: {sorted(missing)}"
        )

    # Parse the actual conformal prediction sets.
    sets = d["conformal_set"].apply(parse_conformal_set)

    # Recalculate coverage directly from the true labels and sets.
    d["covered_recalculated"] = [
        int(true_label) in {int(label) for label in prediction_set}
        for true_label, prediction_set
        in zip(d["true_label"], sets)
    ]

    d["empty"] = sets.apply(len).eq(0)
    d["both"] = sets.apply(len).eq(2)
    d["set_size_recalculated"] = sets.apply(len)

    # Check the saved covered column against our recalculation.
    if "covered" in d.columns:
        saved_covered = d["covered"].astype(bool)

        if not saved_covered.equals(d["covered_recalculated"]):
            raise ValueError(
                f"Coverage mismatch in {file.name}"
            )

    # Calculate metrics separately for each true class.
    for (dataset, seed, model, true_class), group in d.groupby(
        ["dataset", "seed", "model", "true_label"]
    ):
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "model": model,
            "true_class": true_class,
            "n_test": len(group),
            "coverage": round(
                group["covered_recalculated"].mean(), 3
            ),
            "average_set_size": round(
                group["set_size_recalculated"].mean(), 3
            ),
            "empty_rate": round(
                group["empty"].mean(), 3
            ),
            "both_rate": round(
                group["both"].mean(), 3
            ),
        })


result = pd.DataFrame(rows)

result = result.sort_values(
    ["dataset", "seed", "model", "true_class"]
)

output_file = OUT / "per_class_summary.csv"

result.to_csv(output_file, index=False)

print("\nPer-class conformal prediction summary:\n")
print(result.to_string(index=False))

print(f"\nSaved summary to: {output_file}")
print(f"Prediction files processed: {len(prediction_files)}")
print(f"Summary rows generated: {len(result)}")
