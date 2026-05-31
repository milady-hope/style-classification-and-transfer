PY ?= python
DATA ?= data/data500.csv
OUT ?= results

.PHONY: all corpus classify transfer summary

all: corpus classify transfer summary
corpus:
	$(PY) scripts/analyze_corpus.py --data $(DATA) --out-dir $(OUT)
classify:
	$(PY) scripts/classify.py --data $(DATA) --out-dir $(OUT) --grid-search --multi-seed --shap
transfer: classify
	$(PY) scripts/train_rut5.py --data $(DATA) --out-dir $(OUT) --all-configs
summary:
	$(PY) scripts/summarize.py --out-dir $(OUT)