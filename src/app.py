"""
CampusVoice — Streamlit dashboard.

Run with:
    streamlit run src/app.py

This is the presentation-facing UI. Three modes:
  1. "Overview" — sentiment distribution + top aspects + word cloud
     across the entire analyzed dataset.
  2. "By Course" — drill into a single course's feedback.
  3. "Live Demo" — paste in any text, hit Analyze, see GLM classify
     it in real time. This is what we show during the 5-minute pitch.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

# Make sibling modules importable regardless of where streamlit is invoked from
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from glm_client import analyze_feedback  # noqa: E402

DATA_FILE = SRC_DIR.parent / "data" / "feedback_analyzed.csv"

SENTIMENT_COLORS = {
    "positive": "#2E8B57",
    "neutral": "#B0B0B0",
    "negative": "#C0392B",
}

ASPECT_LABELS = {
    "teaching_style": "Teaching Style",
    "course_content": "Course Content",
    "workload": "Workload",
    "materials": "Materials",
    "exams_grading": "Exams & Grading",
    "instructor": "Instructor",
    "logistics": "Logistics",
    "other": "Other",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["aspects_list"] = df["aspects"].apply(lambda s: json.loads(s) if isinstance(s, str) else [])
    df["keywords_list"] = df["keywords"].apply(lambda s: json.loads(s) if isinstance(s, str) else [])
    return df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def sentiment_donut(df: pd.DataFrame):
    counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"]).fillna(0)
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.55,
        color=counts.index,
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(textposition="outside", textinfo="label+percent")
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320)
    return fig


def aspect_bar(df: pd.DataFrame):
    aspect_counts: Counter = Counter()
    for aspects in df["aspects_list"]:
        for a in aspects:
            aspect_counts[a] += 1
    items = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [ASPECT_LABELS.get(a, a) for a, _ in items]
    values = [v for _, v in items]
    fig = px.bar(
        x=values,
        y=labels,
        orientation="h",
        labels={"x": "Mentions", "y": ""},
        color_discrete_sequence=["#2F6FA6"],
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def aspect_sentiment_heatmap(df: pd.DataFrame):
    """Cross-tab: for each aspect, how many positive/neutral/negative mentions?
    This is the "aspect-based sentiment" chart that makes the dashboard feel
    substantial — it's what a professor actually wants to see.
    """
    records = []
    for _, row in df.iterrows():
        for a in row["aspects_list"]:
            records.append({"aspect": ASPECT_LABELS.get(a, a), "sentiment": row["sentiment"]})
    if not records:
        return None
    long = pd.DataFrame(records)
    pivot = long.pivot_table(index="aspect", columns="sentiment", aggfunc="size", fill_value=0)
    for col in ["positive", "neutral", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["positive", "neutral", "negative"]]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    fig = px.bar(
        pivot,
        orientation="h",
        color_discrete_map=SENTIMENT_COLORS,
        labels={"value": "Count", "aspect": "", "sentiment": "Sentiment"},
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380, barmode="stack")
    return fig


def build_wordcloud_image(df: pd.DataFrame, language: str = "all"):
    """Generate a word cloud PNG from the keyword column.

    Note: for Chinese we would normally need a CJK font; we keep both in
    one cloud for simplicity, but separate EN/ZH into two calls when a
    language filter is set.
    """
    subset = df if language == "all" else df[df["language"] == language]
    all_kw: list[str] = []
    for kws in subset["keywords_list"]:
        all_kw.extend(kws)
    if not all_kw:
        return None
    text = " ".join(all_kw)
    wc_kwargs = dict(width=800, height=360, background_color="white", colormap="viridis")
    # Try to pick a CJK-capable font if available (Linux servers often have these)
    if language == "zh":
        for candidate in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]:
            if Path(candidate).exists():
                wc_kwargs["font_path"] = candidate
                break
    wc = WordCloud(**wc_kwargs).generate(text)
    return wc.to_image()


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CampusVoice — Course Feedback Analyzer", page_icon="🎓", layout="wide")

st.markdown(
    """
<style>
  .metric-card { background:#F7F9FC; border:1px solid #E5E9F0; border-radius:10px; padding:14px 18px; }
  .metric-card h3 { margin:0; font-size:13px; color:#556; text-transform:uppercase; letter-spacing:0.5px; }
  .metric-card .val { font-size:30px; font-weight:700; color:#1A2B4A; margin-top:4px; }
  h1 { padding-top:0; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎓 CampusVoice")
st.caption("Smart Course Feedback Analyzer · Powered by GLM-4-Flash")

# Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio("View", ["📊 Overview", "🎯 By Course", "⚡ Live Demo"], label_visibility="collapsed")
    st.markdown("---")
    if DATA_FILE.exists():
        st.success(f"Dataset loaded\n\n`{DATA_FILE.name}`")
    else:
        st.warning("No analyzed data yet. Run `python src/pipeline.py` first.")

# Main ------------------------------------------------------------------
if page == "⚡ Live Demo":
    st.subheader("Paste any student comment — get instant analysis")
    st.write("This hits the GLM-4-Flash API directly. Typical latency: 1–3 seconds.")

    default_text = "The lectures were clear and the professor is really engaged, but the homework load is way too much for a 3-credit course. Exams were fair though."
    text = st.text_area("Review text", value=default_text, height=140)

    if st.button("🔍 Analyze", type="primary"):
        with st.spinner("Calling GLM-4-Flash..."):
            result = analyze_feedback(text)

        col1, col2, col3 = st.columns(3)
        sent = result["sentiment"]
        col1.markdown(
            f"<div class='metric-card'><h3>Sentiment</h3>"
            f"<div class='val' style='color:{SENTIMENT_COLORS.get(sent, '#333')}'>{sent.title()}</div></div>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            f"<div class='metric-card'><h3>Confidence</h3>"
            f"<div class='val'>{result['confidence']:.0%}</div></div>",
            unsafe_allow_html=True,
        )
        col3.markdown(
            f"<div class='metric-card'><h3>Language</h3>"
            f"<div class='val'>{result['language'].upper()}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Aspects detected:** " + ", ".join(ASPECT_LABELS.get(a, a) for a in result["aspects"]))
        st.markdown("**Keywords:** " + ", ".join(f"`{k}`" for k in result["keywords"]))

        with st.expander("Raw JSON"):
            st.json(result)
    st.stop()


# Overview and By-Course both need data
if not DATA_FILE.exists():
    st.info("Run the data pipeline first:\n\n```bash\npython src/generate_data.py --count 300\npython src/pipeline.py\n```")
    st.stop()

df = load_data(DATA_FILE)

if page == "🎯 By Course":
    courses = sorted(df["course_name"].dropna().unique().tolist())
    chosen = st.selectbox("Select a course", courses)
    df_view = df[df["course_name"] == chosen]
else:
    df_view = df

# KPI row ---------------------------------------------------------------
total = len(df_view)
pos_pct = (df_view["sentiment"] == "positive").mean() * 100 if total else 0
neg_pct = (df_view["sentiment"] == "negative").mean() * 100 if total else 0
avg_conf = df_view["confidence"].mean() if total else 0

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='metric-card'><h3>Total Reviews</h3><div class='val'>{total}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='metric-card'><h3>Positive</h3><div class='val' style='color:{SENTIMENT_COLORS['positive']}'>{pos_pct:.0f}%</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='metric-card'><h3>Negative</h3><div class='val' style='color:{SENTIMENT_COLORS['negative']}'>{neg_pct:.0f}%</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='metric-card'><h3>Avg Confidence</h3><div class='val'>{avg_conf:.0%}</div></div>", unsafe_allow_html=True)

st.markdown("")

# Charts row ------------------------------------------------------------
c1, c2 = st.columns([1, 1.2])
with c1:
    st.subheader("Sentiment Distribution")
    st.plotly_chart(sentiment_donut(df_view), use_container_width=True)
with c2:
    st.subheader("Top Aspects Mentioned")
    st.plotly_chart(aspect_bar(df_view), use_container_width=True)

st.subheader("Sentiment by Aspect")
st.caption("What are students actually upset or happy about?")
heatmap = aspect_sentiment_heatmap(df_view)
if heatmap is not None:
    st.plotly_chart(heatmap, use_container_width=True)

# Word cloud ------------------------------------------------------------
st.subheader("Keyword Cloud")
wc_lang = st.radio("Language", ["all", "en", "zh"], horizontal=True, label_visibility="collapsed")
img = build_wordcloud_image(df_view, wc_lang)
if img is not None:
    st.image(img, use_container_width=True)
else:
    st.info("Not enough keywords in this subset.")

# Sample rows -----------------------------------------------------------
st.subheader("Sample Reviews")
st.dataframe(
    df_view[["review_id", "course_name", "language", "sentiment", "confidence", "review_text"]]
    .head(15)
    .reset_index(drop=True),
    use_container_width=True,
)
