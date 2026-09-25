
# Reproducing the results

Everything runs from the repository root. The data stay outside the repository (one
directory up by default; see `docs/DATA.md`). Every stage is cached, so an interrupted run
resumes where it stopped.

## 1. Environment

```bash
python3 --version              # 3.11 or later (3.12 used)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
ffmpeg -version                # needed by Whisper
```

A GPU is optional. The medium model decodes a few thousand short recordings in a few hours
on one GPU and roughly ten times longer on a CPU.

## 2. Check the code without data

```bash
make test        # synthetic transcripts, a few seconds
make lint
```

## 3. Index and transcribe

```bash
make index                            # outputs/trial_index.csv, keys/word_lists.json
make transcribe MODEL=medium          # cache/transcripts/<basename>.json
make transcribe-words MODEL=medium    # cache/transcripts_words/ (word probabilities and onsets)
```

`src/transcribe.py` refuses to overwrite a cached transcript produced by a larger model.
Results are pinned to the production cache, so a fresh decode should go to a new directory
(`SWC_TRANSCRIPT_CACHE=...`) and be compared with `tools/pipeline/compare_transcript_caches.py`.

## 4. Teacher-forced alternatives (candidate rescue, R3)

```bash
make alternatives MODEL=medium
```

Writes `outputs/word_alternatives_rescue.csv` for the recordings listed in
`outputs/run_final/alternatives_targets.txt` (resumable). `tools/pipeline/run_final.py`
builds that list.

## 5. Score with the frozen recipe

```bash
make score                     # scores_word_recall.csv, scores_serial.csv, recall_tokens_long.csv
cat outputs/recipe_manifest.json
```

`SWC_RECIPE_CONFIG=automatic make score` gives the configuration without the confusion map.
Serial position needs `keys/word_lists_ordered.json` (see `keys/README.md`).

## 6. Development-sample report and regression guard

```bash
make validate                  # calibration + docs/results/validation_report.md and .json
make check                     # recomputes every report number, tolerance 0.002
```

Needs the manual score file (`HAND_MEM_CSV`), the manual transcription workbook
(`HUMAN_XLSX`), and `keys/locked_participants.txt` (see `config.py`).

## 7. Validation sample

```bash
python3 tools/pipeline/run_final.py --dry-run     # lists the steps and checks the inputs
make study2 PASS_A=<Rater 1 workbook> PASS_B=<Rater 2 workbook>
```

`--dry-run` on `study2_reliability.py` validates the two workbooks and prints the n per
comparison without writing. The aggregates from the reported run are in `docs/results/study2/`.

## 8. Exploratory revision

```bash
SWC_RECIPE_VERSION=2026-09-v2 python3 src/score_wordrecall.py --out-dir outputs/v2
SWC_RECIPE_VERSION=2026-09-v2 python3 src/score_serial.py --out-dir outputs/v2
python3 tools/pipeline/dev_agreement_v2.py --scores outputs/v2/scores_word_recall.csv
```

## 9. What to compare against

| File | Content |
|---|---|
| `docs/results/validation_report.md` | development-sample tables, thresholds, manifest |
| `docs/results/study2/study2_report.md` | validation-sample tables and data-quality checks |
| `docs/results/study2/study2_results.json` | the same numbers, machine readable |
| `docs/results/study2_v2_exploratory.md` | exploratory revision and its disclosure |

The frozen score tables used in the paper have content hashes beginning `f818f63e`
(`scores_word_recall.csv`) and `be40fe4e` (`scores_serial.csv`).
