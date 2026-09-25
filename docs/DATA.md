
# Data

No participant data are in this repository. This page lists what the pipeline reads and
writes and where each item lives. Every path is set in `config.py` and can be overridden
with the environment variable named there.

## Layout

```
project root/                        SWC_PROJECT_ROOT (default: the parent of the repository)
  <audio folder>/                    raw .m4a recordings (SWC_AUDIO_ROOT)
  <watch export>.zip                 .xlsx export with the memory-task sheet (SWC_PROMPTS_ZIP, SWC_MEM_SHEET)
  <manual scores>.csv                research-assistant recall scores (SWC_HAND_MEM_CSV)
  <manual transcripts>.xlsx          verified manual transcripts, tab "Memory Recall" (SWC_HUMAN_XLSX)
  <rating workbooks>.xlsx            validation sample, one workbook per rater
  ARTS/                              this repository
    keys/                            word lists and the held-out list (not tracked; see keys/README.md)
    cache/                           Whisper transcripts (not tracked)
    outputs/                         score tables and working files (not tracked)
    docs/results/                    aggregate reports (tracked)
```

## Inputs

| Item | Format | Used by |
|---|---|---|
| Recordings | `.m4a`, one per recall trial; the file name carries the watch code, task, date and time | `io_index.py`, `transcribe.py`, `word_alternatives.py` |
| Watch export | `.zip` of `.xlsx` sheets; the memory sheet lists each prompt, its list and its audio file | `io_index.py` (writes `keys/word_lists.json`) |
| Presentation order | `keys/word_lists_ordered.json` | `score_serial.py` |
| Held-out participants | `keys/locked_participants.txt` or `SWC_LOCKED_PARTICIPANTS` | `run_final.py`, `validate_recipe.py`, `dev_agreement_v2.py` |
| Manual recall scores | `.csv` with Correct, Repetitions, Intrusions, Primacy, Middle, Recency (header on the second row) | `validate_recipe.py`, `validate_rescue.py` |
| Manual transcripts | `.xlsx`, tab "Memory Recall" with ID, Date, Time and Audio Transcription | `confidence_calibration.py`, `validate.py` |
| Rating workbooks | two `.xlsx` files with "Transcriptions" and "Score - word recall" tabs | `study2_reliability.py` |

## Outputs

| Item | Location | Tracked |
|---|---|---|
| Trial index | `outputs/trial_index.csv` | no |
| Transcripts, production and word-timestamp | `cache/transcripts/`, `cache/transcripts_words/` | no |
| Teacher-forced alternatives | `outputs/word_alternatives*.csv` | no |
| Score tables | `outputs/scores_word_recall.csv`, `outputs/scores_serial.csv` | no |
| Per-word audit table | `outputs/recall_tokens_long.csv` | no |
| Recipe manifest | `outputs/recipe_manifest.json` | no (copied into the reports) |
| Development-sample report | `docs/results/validation_report.md`, `.json` | yes (aggregates) |
| Validation-sample report | `docs/results/study2/` | yes (aggregates) |

## Privacy

Recordings are speech and can identify a person; they never enter the repository.
Transcripts, per-recording tables, word lists and participant identifiers are gitignored,
and the tracked reports contain only aggregates.
