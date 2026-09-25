# Entry points for the frozen pipeline. Every target runs from the repository root;
# data paths come from config.py (override with SWC_* variables).
PY ?= python3
MODEL ?= medium
PASS_A ?= ../rating_workbook_rater1.xlsx
PASS_B ?= ../rating_workbook_rater2.xlsx

.PHONY: help index transcribe transcribe-words alternatives score calibration validate check study2 test lint clean-pyc

help:
	@echo "index            trial index and word lists from the watch export"
	@echo "transcribe       Whisper $(MODEL) production decode -> cache/transcripts/"
	@echo "transcribe-words Whisper $(MODEL) word-timestamp decode -> cache/transcripts_words/"
	@echo "alternatives     teacher-forced alternatives table (candidate rescue, R3)"
	@echo "score            frozen recipe scores, serial position and the per-word audit table"
	@echo "calibration      word-confidence calibration against manual transcripts"
	@echo "validate         development-sample report -> docs/results/validation_report.md"
	@echo "check            regression guard: recompute the report numbers (tolerance 0.002)"
	@echo "study2           validation-sample agreement (PASS_A=... PASS_B=...) -> outputs/study2/"
	@echo "test             pytest on synthetic transcripts (no data needed)"
	@echo "lint             ruff"

index:
	$(PY) src/io_index.py

transcribe:
	$(PY) src/transcribe.py --model $(MODEL)

transcribe-words:
	SWC_TRANSCRIPT_CACHE=cache/transcripts_words $(PY) src/transcribe.py --model $(MODEL) --word-timestamps

alternatives:
	$(PY) tools/pipeline/word_alternatives.py --targets outputs/run_final/alternatives_targets.txt --model $(MODEL) --out outputs/word_alternatives_rescue.csv

score:
	$(PY) src/score_wordrecall.py
	$(PY) src/score_serial.py
	$(PY) tools/pipeline/export_token_classification.py

calibration:
	$(PY) tools/pipeline/confidence_calibration.py

validate: calibration
	$(PY) tools/pipeline/validate_recipe.py

check:
	$(PY) tools/pipeline/validate_recipe.py --check

study2:
	$(PY) tools/pipeline/study2_reliability.py --pass-a $(PASS_A) --pass-b $(PASS_B)

test:
	$(PY) -m pytest -q -p no:cacheprovider

lint:
	ruff check src tools tests

clean-pyc:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
