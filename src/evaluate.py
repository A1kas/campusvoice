"""
Model evaluation.

Because we asked GLM to generate reviews with a target sentiment, we have
a pseudo-ground-truth label. Comparing predicted sentiment against
``intended_sentiment`` gives us an accuracy number and a confusion matrix
we can put in the final slides.

Caveat noted in the report: this measures consistency of GLM with itself,
not human-labeled ground truth. A proper eval would use hand-annotated
reviews — see "Future Work" in the report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def evaluate(analyzed_csv: Path) -> None:
    df = pd.read_csv(analyzed_csv)
    df = df[df["intended_sentiment"].notna() & (df["intended_sentiment"] != "")]
    if len(df) == 0:
        print("No rows with intended_sentiment labels — cannot evaluate.")
        return

    y_true = df["intended_sentiment"].astype(str)
    y_pred = df["sentiment"].astype(str)

    labels = ["positive", "neutral", "negative"]

    print(f"\nEvaluating on {len(df)} reviews\n")
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("\nConfusion Matrix (rows = true, cols = predicted)")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    print(cm_df.to_string())

    # Save for the slides
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    (OUTPUT_DIR / "classification_report.txt").write_text(report_str, encoding="utf-8")

    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {accuracy:.3f}")
    (OUTPUT_DIR / "accuracy.txt").write_text(f"{accuracy:.4f}\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_DIR / "feedback_analyzed.csv")
    args = parser.parse_args()
    evaluate(args.input)
