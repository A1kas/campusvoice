"""
GLM API client wrapper.

All calls to Zhipu AI's GLM-4-Flash model route through here. This module
handles: API key loading, retry on transient errors, JSON-response parsing,
and simple on-disk caching so we don't re-pay for the same prompt during
development.

The ``analyze_feedback`` function is the one the pipeline actually calls —
it sends a single student review and returns structured sentiment + aspect
tags + keywords.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "glm-4-flash"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".glm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------
# zhipuai is imported lazily inside get_client() so that the rest of this
# module (cache helpers, JSON parsing, prompt templates) can be imported
# and unit-tested without the SDK or network access present.
_client: Any | None = None


def get_client() -> Any:
    """Return a lazily-initialised ZhipuAI client.

    Reads ``ZHIPUAI_API_KEY`` from environment (or .env file). Raises a
    clear error if the key or the SDK is missing — common cause of silent
    failures in student projects.
    """
    global _client
    if _client is None:
        try:
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise RuntimeError(
                "zhipuai SDK not installed. Run: pip install -r requirements.txt"
            ) from e
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ZHIPUAI_API_KEY not found. Create a .env file in the project "
                "root with: ZHIPUAI_API_KEY=your_key_here"
            )
        _client = ZhipuAI(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
def _cache_key(prompt: str, system: str) -> str:
    h = hashlib.sha256((system + "||" + prompt).encode("utf-8")).hexdigest()
    return h[:16]


def _cache_get(key: str) -> dict[str, Any] | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _cache_put(key: str, value: dict[str, Any]) -> None:
    p = CACHE_DIR / f"{key}.json"
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Low-level call
# ---------------------------------------------------------------------------
def chat(
    prompt: str,
    system: str = "You are a helpful assistant.",
    *,
    temperature: float = 0.2,
    max_retries: int = 3,
    use_cache: bool = True,
) -> str:
    """Send a single-turn chat and return the raw string response.

    Retries on transient failures with exponential backoff. Cache keyed on
    ``(system, prompt)``; temperature is not part of the key because the
    caller is responsible for deciding whether determinism matters for
    caching. For our NLP pipeline we set temperature=0.2 and cache freely.
    """
    key = _cache_key(prompt, system)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached["content"]

    client = get_client()
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            content = resp.choices[0].message.content or ""
            if use_cache:
                _cache_put(key, {"content": content})
            return content
        except Exception as e:  # noqa: BLE001 — retry on anything transient
            last_err = e
            wait = 2**attempt
            print(f"[glm_client] attempt {attempt + 1} failed: {e} — retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"GLM call failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Structured JSON extraction
# ---------------------------------------------------------------------------
def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # drop leading ```json\n or ```\n
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def chat_json(
    prompt: str,
    system: str = "You are a helpful assistant that responds only with valid JSON.",
    **kwargs: Any,
) -> dict[str, Any]:
    """Call chat() and parse the response as JSON.

    Tolerates markdown code fences (```json ... ```) which GLM sometimes
    emits despite instructions. If parsing fails, returns an error dict
    rather than raising — pipeline code can filter these out and keep going.
    """
    raw = chat(prompt, system=system, **kwargs)
    try:
        return json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError as e:
        return {"_error": "json_parse_failed", "_raw": raw, "_msg": str(e)}


# ---------------------------------------------------------------------------
# Task-specific helpers
# ---------------------------------------------------------------------------
SENTIMENT_SYSTEM = """You are a precise NLP assistant analyzing student course feedback.
You must respond with ONLY a valid JSON object and no other text.
Do not use markdown code fences. Do not add commentary."""

SENTIMENT_PROMPT_TEMPLATE = """Analyze this student course review and return JSON with these exact keys:

- "sentiment": one of "positive", "negative", "neutral"
- "confidence": float between 0.0 and 1.0
- "aspects": list of 1-3 strings from this fixed set only: ["teaching_style", "course_content", "workload", "materials", "exams_grading", "instructor", "logistics", "other"]
- "keywords": list of 2-5 short keywords (1-3 words each) extracted from the review. Keep them in the review's original language.
- "language": "en" or "zh"

Review to analyze:
\"\"\"{review}\"\"\"

Respond with only the JSON object."""


def analyze_feedback(review_text: str) -> dict[str, Any]:
    """Analyze a single piece of student feedback.

    Returns a dict with keys: sentiment, confidence, aspects, keywords,
    language. On parse failure, the keys are populated with safe defaults
    so the downstream DataFrame stays well-typed.
    """
    prompt = SENTIMENT_PROMPT_TEMPLATE.format(review=review_text.replace('"""', "'''"))
    result = chat_json(prompt, system=SENTIMENT_SYSTEM)

    if "_error" in result:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "aspects": ["other"],
            "keywords": [],
            "language": "unknown",
            "_error": result["_error"],
        }

    # Defensive normalization — GLM sometimes returns strings instead of lists
    aspects = result.get("aspects", [])
    if isinstance(aspects, str):
        aspects = [aspects]
    keywords = result.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]

    return {
        "sentiment": str(result.get("sentiment", "neutral")).lower(),
        "confidence": float(result.get("confidence", 0.0)),
        "aspects": [str(a).lower() for a in aspects],
        "keywords": [str(k) for k in keywords],
        "language": str(result.get("language", "unknown")).lower(),
    }
