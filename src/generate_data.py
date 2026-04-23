"""
Synthetic feedback generator.

Uses GLM-4-Flash to generate realistic-looking student course reviews for
the demo. This is deliberately a one-shot script, not part of the runtime
pipeline — run once, commit the resulting CSV, reuse for all demos.

Why synthesize instead of scraping? Two reasons:
  1. Ground-truth labels: we ask the model to also produce the sentiment
     label it intended, giving us a labeled dataset for evaluation.
  2. Controlled distribution: we can force coverage of every aspect
     category and a realistic positive/neutral/negative mix.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from glm_client import chat_json

OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "feedback_raw.csv"

COURSES = [
    ("CS201", "Data Structures", "Computer Science"),
    ("CS310", "Operating Systems", "Computer Science"),
    ("CS330", "Database Systems", "Computer Science"),
    ("MATH240", "Linear Algebra", "Mathematics"),
    ("ENG150", "Academic Writing", "English"),
    ("BUS220", "Introduction to Marketing", "Business"),
    ("PHY101", "General Physics I", "Physics"),
    ("CS420", "Machine Learning", "Computer Science"),
]

ASPECTS = [
    "teaching_style",
    "course_content",
    "workload",
    "materials",
    "exams_grading",
    "instructor",
    "logistics",
]

SENTIMENT_MIX = {"positive": 0.45, "negative": 0.30, "neutral": 0.25}


GENERATE_SYSTEM = """You are generating synthetic training data for a university course
feedback analyzer. Write realistic, diverse student reviews in the requested language.
Respond with ONLY a valid JSON object. No code fences, no commentary."""


def build_prompt(course: str, department: str, sentiment: str, aspect: str, lang: str) -> str:
    lang_name = "English" if lang == "en" else "Simplified Chinese"
    return f"""Generate ONE realistic anonymous student review for the course "{course}" ({department}).

Constraints:
- Language: {lang_name}
- Overall sentiment: {sentiment}
- The review should primarily touch on this aspect: {aspect}
- Length: 1-3 sentences, 20-60 words
- Sound like a real undergraduate, not a marketing copywriter
- Can include specific details (homework load, exam difficulty, TA quality, etc.)
- Do NOT use the words "positive", "negative", or "neutral" in the review itself
- Do NOT mention the course code

Respond with JSON:
{{"review": "<the review text>", "intended_sentiment": "{sentiment}", "intended_aspect": "{aspect}"}}
"""


def pick_sentiment() -> str:
    r = random.random()
    cum = 0.0
    for s, p in SENTIMENT_MIX.items():
        cum += p
        if r <= cum:
            return s
    return "neutral"


def generate_one(course_code: str, course_name: str, department: str, lang: str) -> dict[str, Any] | None:
    sentiment = pick_sentiment()
    aspect = random.choice(ASPECTS)
    prompt = build_prompt(course_name, department, sentiment, aspect, lang)
    result = chat_json(prompt, system=GENERATE_SYSTEM, temperature=0.9, use_cache=False)
    if "_error" in result or "review" not in result:
        return None
    return {
        "course_code": course_code,
        "course_name": course_name,
        "department": department,
        "language": lang,
        "review_text": result["review"].strip(),
        "intended_sentiment": result.get("intended_sentiment", sentiment),
        "intended_aspect": result.get("intended_aspect", aspect),
    }


def main(count: int, zh_ratio: float) -> None:
    random.seed(42)
    rows: list[dict[str, Any]] = []
    target_zh = int(count * zh_ratio)

    for i in range(count):
        course_code, course_name, department = random.choice(COURSES)
        lang = "zh" if i < target_zh else "en"
        row = generate_one(course_code, course_name, department, lang)
        if row is None:
            print(f"  [{i + 1}/{count}] SKIPPED (generation failed)")
            continue
        rows.append(row)
        print(f"  [{i + 1}/{count}] {lang} | {row['intended_sentiment']:<8} | {row['review_text'][:60]}...")

    random.shuffle(rows)  # mix languages and courses

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "review_id",
                "course_code",
                "course_name",
                "department",
                "language",
                "review_text",
                "intended_sentiment",
                "intended_aspect",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow({"review_id": f"R{idx:04d}", **row})

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300, help="number of reviews to generate")
    parser.add_argument("--zh-ratio", type=float, default=0.35, help="fraction in Chinese")
    args = parser.parse_args()
    main(args.count, args.zh_ratio)
