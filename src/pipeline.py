"""
Analysis pipeline.

Takes raw feedback CSV → runs each review through GLM → writes an
enriched CSV with sentiment labels, aspect tags, keywords, and confidence.

Designed to be idempotent: if output already exists, skip rows already
processed (matched by review_id). This lets us re-run after a crash
without re-paying for every call.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from glm_client import analyze_feedback

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "feedback_raw.csv"
DEFAULT_OUTPUT = DATA_DIR / "feedback_analyzed.csv"


def load_already_done(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        prev = pd.read_csv(output_path)
        return set(prev["review_id"].astype(str).tolist())
    except Exception:  # noqa: BLE001
        return set()


def run(input_path: Path, output_path: Path, limit: int | None = None) -> None:
    df_in = pd.read_csv(input_path)
    if limit is not None:
        df_in = df_in.head(limit)

    done = load_already_done(output_path)
    todo = df_in[~df_in["review_id"].astype(str).isin(done)]
    print(f"Input: {len(df_in)} rows | already done: {len(done)} | to process: {len(todo)}")

    if len(todo) == 0:
        print("Nothing to do.")
        return

    rows_out: list[dict] = []
    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        analysis = analyze_feedback(row["review_text"])
        out_row = {
            "review_id": row["review_id"],
            "course_code": row.get("course_code", ""),
            "course_name": row.get("course_name", ""),
            "department": row.get("department", ""),
            "language": analysis["language"],
            "review_text": row["review_text"],
            "sentiment": analysis["sentiment"],
            "confidence": analysis["confidence"],
            "aspects": json.dumps(analysis["aspects"], ensure_ascii=False),
            "keywords": json.dumps(analysis["keywords"], ensure_ascii=False),
            "intended_sentiment": row.get("intended_sentiment", ""),
            "intended_aspect": row.get("intended_aspect", ""),
        }
        rows_out.append(out_row)
        print(f"  [{i}/{len(todo)}] {row['review_id']}: {analysis['sentiment']:<8} ({analysis['confidence']:.2f})")

    df_new = pd.DataFrame(rows_out)
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nWrote {len(df_final)} total rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="only process first N rows")
    args = parser.parse_args()
    run(args.input, args.output, args.limit)
