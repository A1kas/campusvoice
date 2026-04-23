.PHONY: install generate analyze eval run clean

install:
	pip install -r requirements.txt

# Generate 300 synthetic reviews
generate:
	python src/generate_data.py --count 300 --zh-ratio 0.35

# Run the full NLP pipeline on the raw data
analyze:
	python src/pipeline.py

# Evaluate prediction vs intended labels
eval:
	python src/evaluate.py

# Launch the Streamlit dashboard
run:
	streamlit run src/app.py

# Full end-to-end: generate → analyze → eval → dashboard
all: generate analyze eval run

clean:
	rm -rf data/.glm_cache
	rm -f data/feedback_raw.csv data/feedback_analyzed.csv
	rm -rf outputs/*
