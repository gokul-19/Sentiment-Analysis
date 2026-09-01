"""Train, evaluate, and persist the best available sentiment model."""

import json
from pathlib import Path

from sentiment_model import DEFAULT_DATASET, DEFAULT_MODEL, load_dataset, save_model, train_and_compare


def main() -> None:
    data = load_dataset(DEFAULT_DATASET)
    model, comparison, metadata = train_and_compare(data)
    save_model(model, DEFAULT_MODEL)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    (reports / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"\nSaved model: {DEFAULT_MODEL}")
    print(f"Saved reports: {reports.resolve()}")


if __name__ == "__main__":
    main()
