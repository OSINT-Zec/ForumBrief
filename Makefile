# ---- Paths ----
PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
ENV    := .venv

# ---- Files ----
RAW     := data/raw.jsonl
DEDUP   := data/dedup.jsonl
CLUST   := data/clusters.json
SUMMARY := out/summary.md
TIPS    := out/tips.csv

# ---- Default ----
.DEFAULT_GOAL := help

# ---- Phony ----
.PHONY: help venv install check-env collect clean dedupe cluster summarize export all run wipe

help:
	@echo "Targets:"
	@echo "  make venv         - create venv"
	@echo "  make install      - install requirements"
	@echo "  make check-env    - verify .env and config.yaml"
	@echo "  make collect      - one-time fetch from Reddit/StackExchange"
	@echo "  make clean        - normalize & strip noise"
	@echo "  make dedupe       - semantic near-duplicate removal"
	@echo "  make cluster      - embed + cluster + theme label"
	@echo "  make summarize    - LLM map-reduce summaries"
	@echo "  make export       - write summary.md and tips.csv"
	@echo "  make all|run      - run everything (collect → export)"
	@echo "  make wipe         - delete data/out artifacts"

venv:
	python -m venv .venv

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

check-env:
	@test -f .env || (echo "Missing .env. Copy .env.example -> .env and fill keys."; exit 1)
	@test -f config.yaml || (echo "Missing config.yaml. Create it from the template in this readme."; exit 1)
	@mkdir -p data out

collect: check-env
	$(PYTHON) pipeline.py collect --config config.yaml --out $(RAW)

clean: check-env
	$(PYTHON) pipeline.py clean --inp $(RAW) --out data/clean.jsonl

dedupe: check-env
	$(PYTHON) pipeline.py dedupe --inp data/clean.jsonl --out $(DEDUP)

cluster: check-env
	$(PYTHON) pipeline.py cluster --inp $(DEDUP) --out $(CLUST)

summarize: check-env
	$(PYTHON) pipeline.py summarize --inp $(DEDUP) --clusters $(CLUST) --out out/cluster_summaries.json

export: check-env
	$(PYTHON) pipeline.py export --cluster_summ out/cluster_summaries.json --clusters $(CLUST) --inp $(DEDUP) --md $(SUMMARY) --csv $(TIPS)

all: install run
run: collect clean dedupe cluster summarize export

wipe:
	rm -rf data/*.jsonl data/*.json out/*.md out/*.csv out/*.json
