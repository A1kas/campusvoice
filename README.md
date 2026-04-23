# 🎓 CampusVoice — Smart Course Feedback Analyzer

> M8 Final Project · Applied NLP for university course evaluation
> **Team:** Miao Jingzhe · Zeng · Zhong · Liang

An end-to-end NLP pipeline that turns unstructured student course reviews into
actionable insights for professors and administrators. Built on **Zhipu AI's
GLM-4-Flash** model — multilingual (English + Chinese), free tier, sub-second
latency per review.

---

## What it does

Feed in a CSV of anonymous student course reviews. Out comes:

- **Sentiment classification** — positive / neutral / negative, with confidence
- **Aspect-based tagging** — which part of the course is being discussed
  (teaching style, workload, materials, exams, instructor, logistics)
- **Keyword extraction** — the phrases students actually use
- **An interactive dashboard** — filterable by course, language, sentiment

Plus a live-demo mode for the final presentation: paste any comment, watch GLM
classify it in real time.

---

## Repo layout

```
campusvoice/
├── src/
│   ├── glm_client.py      # GLM-4-Flash wrapper (retry, cache, JSON parsing)
│   ├── generate_data.py   # Synthesize 300 labeled reviews for the demo
│   ├── pipeline.py        # Run each review through GLM → enriched CSV
│   ├── evaluate.py        # Accuracy + confusion matrix vs intended labels
│   └── app.py             # Streamlit dashboard
├── data/
│   ├── feedback_raw.csv       # generated
│   ├── feedback_analyzed.csv  # pipeline output
│   └── .glm_cache/            # response cache (auto, gitignored)
├── outputs/                   # evaluation artifacts for the slides
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# edit .env, paste your GLM key
```

Get a key at [open.bigmodel.cn](https://open.bigmodel.cn/). GLM-4-Flash is free.

### 3. Generate synthetic data (first run only, ~3 minutes)

```bash
make generate
```

This asks GLM to write 300 realistic student reviews — a mix of positive,
negative, and neutral, in both English and Chinese, across 8 courses. Each
review comes with a target sentiment label we use later for evaluation.

### 4. Run the NLP pipeline (~5 minutes)

```bash
make analyze
```

Every review gets classified. Results are cached on disk — re-running is free.

### 5. Evaluate

```bash
make eval
```

Prints accuracy and a confusion matrix. Writes artifacts to `outputs/`.

### 6. Launch the dashboard

```bash
make run
```

Opens at `http://localhost:8501`. Three views:

- **Overview** — all courses, all feedback
- **By Course** — pick one course, see only its reviews
- **Live Demo** — paste any text, analyze it live (this is the presentation mode)

---

## Division of work

| Member | Role | Modules |
|--------|------|---------|
| Miao | Data Engineering & Preprocessing | `generate_data.py`, `pipeline.py` |
| Zeng | NLP Integration & Model Deployment | `glm_client.py`, prompt design |
| Zhong | Frontend UI & Data Visualization | `app.py`, chart design |
| Liang | Project Management & Testing | `evaluate.py`, README, demo script |

---

## Design decisions worth calling out

**Why GLM-4-Flash, not a fine-tuned classifier?**
A fine-tuned BERT would need labeled data we don't have and a training loop
we don't have time for. GLM-4-Flash is free, handles Chinese + English out
of the box, and lets us do both sentiment classification *and* aspect tagging
in a single API call via structured JSON output.

**Why cache every response?**
During development we re-run the pipeline dozens of times. Caching on a
SHA-256 of `(system, prompt)` means we only pay (in latency and quota) for
each unique review once. Purge with `make clean`.

**Why synthesize the data instead of scraping real reviews?**
Two reasons: (1) it gives us ground-truth labels for evaluation — we asked
GLM to write a review with a *target* sentiment, then check whether it
classifies its own output correctly. (2) We control the distribution, so
every aspect category is represented.

This is documented as a limitation in the report. Real deployment would
use human-annotated data.

---

## For the live demo (5-minute pitch)

1. Open the dashboard on **Overview** — show the KPIs and sentiment-by-aspect chart.
2. Switch to **By Course** — pick a course with mixed feedback, narrate what a professor would learn from this view.
3. Switch to **Live Demo** — paste a review the audience suggests, or a pre-prepared bilingual one. Show the JSON output.
4. Close with accuracy number from `make eval`.
