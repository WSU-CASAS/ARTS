# Tools

Run from the repository root. Paths come from `config.py`.

## pipeline/

| Script | What it does |
|---|---|
| `run_final.py` | Driver for the full run: index, transcribe, word-timestamp decode, alternatives, score, regression guard, validation-sample agreement. `--dry-run` lists the steps and checks the inputs. |
| `word_alternatives.py` | Teacher-forced pass that records each word's probability, entropy and top alternatives (input to candidate rescue, R3). |
| `export_token_classification.py` | Per-word audit table: every spoken token with its classification and the rule that touched it. |
| `confidence_calibration.py` | Word-confidence calibration and transcription accuracy against verified manual transcripts (Supplement S1). |
| `validate_recipe.py` | Development-sample report (`docs/results/validation_report.md`); `--check` is the regression guard. |
| `study2_reliability.py` | Agreement of the pipeline with each rater and between raters, plus word error rates (`docs/results/study2/`). |
| `compare_transcript_caches.py` | Compares two transcript caches: text differences and the effect on scores (Supplement S1). |
| `dev_agreement_v2.py` | Development-sample agreement of the exploratory recipe against the frozen one. |

## analysis/

| Script | What it does |
|---|---|
| `drift_control_analysis.py` | Separates the effect of the word-timestamp setting from run-to-run variation (Supplement S1). |
| `validate_rescue.py` | Candidate rescue against the manual scores across probability floors and entropy gates (Supplement S3). |
