# Study 2: reliability against two blind human passes

Generated 2026-09-25. Machine scores: `scores_word_recall.csv`; recipe recipe_version 2026-09-freeze, recipe_config ra_informed.

Aggregate tables only. Per-clip pairs are written to CSVs beside the report in outputs/, which is not tracked.

## Inputs and data quality

| Item | Pass A | Pass B |
|---|---|---|
| Transcriptions: rows with an audio file | 585 | 585 |
| Transcriptions: rows untranscribed | 0 | 0 |
| Transcriptions: duplicate audio files | 0 | 0 |
| Transcriptions: unrecognised task label | 0 | 0 |
| Score - word recall: rows with an audio file | 585 | 585 |
| Score - word recall: rows not yet scored (all cells blank) | 0 | 2 |
| Score - word recall: duplicate audio files | 0 | 0 |
| Score - word recall / Correct: blank cells in scored rows | 0 | 0 |
| Score - word recall / Repetitions: blank cells in scored rows | 0 | 0 |
| Score - word recall / Intrusions: blank cells in scored rows | 0 | 0 |
| Score - word recall / Primacy: blank cells in scored rows | 0 | 0 |
| Score - word recall / Middle: blank cells in scored rows | 0 | 0 |
| Score - word recall / Recency: blank cells in scored rows | 0 | 0 |
| Score - word recall / Correct: non-numeric cells | 0 | 0 |
| Score - word recall / Repetitions: non-numeric cells | 0 | 0 |
| Score - word recall / Intrusions: non-numeric cells | 0 | 0 |
| Score - word recall / Primacy: non-numeric cells | 0 | 0 |
| Score - word recall / Middle: non-numeric cells | 0 | 0 |
| Score - word recall / Recency: non-numeric cells | 0 | 0 |

Machine transcript found for 585 of 585 clips transcribed by at least one pass.

## Matching

| Measure | Scored by either pass | With machine row | Without machine row | Pass A only | Pass B only |
|---|---|---|---|---|---|
| Recall correct | 585 | 585 | 0 | 2 | 0 |
| Recall repetitions | 585 | 585 | 0 | 2 | 0 |
| Recall intrusions | 585 | 585 | 0 | 2 | 0 |
| Primacy | 585 | 585 | 0 | 2 | 0 |
| Middle | 585 | 585 | 0 | 2 | 0 |
| Recency | 585 | 585 | 0 | 2 | 0 |

## Score agreement

ICC(2,1) is two-way random effects, single rater, absolute agreement (validate.agreement_stats). Bias is the second member of the comparison minus the first: machine minus human, or pass B minus pass A.

| Measure | Comparison | n | ICC(2,1) | MAE | Bias | Exact | Within 1 |
|---|---|---|---|---|---|---|---|
| Recall correct | machine vs pass A | 585 | 0.971 | 0.21 | 0.01 | 0.83 | 0.97 |
| Recall correct | machine vs pass B | 583 | 0.968 | 0.21 | -0.00 | 0.83 | 0.97 |
| Recall correct | pass A vs pass B | 583 | 0.987 | 0.11 | 0.01 | 0.91 | 0.98 |
| Recall correct | machine vs mean of A and B | 583 | 0.973 | 0.21 | 0.00 | 0.80 | 0.97 |
| Recall repetitions | machine vs pass A | 585 | 0.079 | 0.58 | 0.56 | 0.83 | 0.89 |
| Recall repetitions | machine vs pass B | 583 | 0.086 | 0.58 | 0.57 | 0.83 | 0.89 |
| Recall repetitions | pass A vs pass B | 583 | 0.781 | 0.04 | -0.00 | 0.97 | 0.99 |
| Recall repetitions | machine vs mean of A and B | 583 | 0.083 | 0.58 | 0.57 | 0.81 | 0.89 |
| Recall intrusions | machine vs pass A | 585 | 0.175 | 0.66 | 0.17 | 0.69 | 0.88 |
| Recall intrusions | machine vs pass B | 583 | 0.192 | 0.65 | 0.15 | 0.70 | 0.88 |
| Recall intrusions | pass A vs pass B | 583 | 0.891 | 0.15 | 0.03 | 0.88 | 0.98 |
| Recall intrusions | machine vs mean of A and B | 583 | 0.186 | 0.65 | 0.16 | 0.65 | 0.89 |
| Primacy | machine vs pass A | 585 | 0.963 | 0.07 | 0.01 | 0.94 | 0.99 |
| Primacy | machine vs pass B | 583 | 0.968 | 0.06 | -0.01 | 0.95 | 0.99 |
| Primacy | pass A vs pass B | 583 | 0.984 | 0.04 | 0.01 | 0.96 | 1.00 |
| Primacy | machine vs mean of A and B | 583 | 0.969 | 0.06 | 0.00 | 0.92 | 0.99 |
| Middle | machine vs pass A | 585 | 0.954 | 0.12 | -0.01 | 0.90 | 0.99 |
| Middle | machine vs pass B | 583 | 0.950 | 0.13 | -0.00 | 0.88 | 0.99 |
| Middle | pass A vs pass B | 583 | 0.970 | 0.08 | -0.00 | 0.93 | 0.99 |
| Middle | machine vs mean of A and B | 583 | 0.959 | 0.12 | -0.00 | 0.86 | 0.99 |
| Recency | machine vs pass A | 585 | 0.974 | 0.05 | 0.00 | 0.95 | 1.00 |
| Recency | machine vs pass B | 583 | 0.973 | 0.05 | 0.00 | 0.95 | 0.99 |
| Recency | pass A vs pass B | 583 | 0.990 | 0.02 | -0.00 | 0.98 | 1.00 |
| Recency | machine vs mean of A and B | 583 | 0.977 | 0.05 | 0.00 | 0.94 | 1.00 |

## Transcript agreement (word error rate)

WER is edit distance over reference words (common.wer). The reference is the human pass named in the comparison (pass A for the human-human row); the mean-of-humans row averages the two machine WERs per clip. Corpus WER pools edits and reference words across clips.

| Task | Comparison | n | Mean WER | Median WER | Corpus WER |
|---|---|---|---|---|---|
| word-list recall | machine vs pass A | 577 | 0.258 | 0.077 | 0.250 |
| word-list recall | machine vs pass B | 576 | 0.307 | 0.000 | 0.267 |
| word-list recall | pass A vs pass B | 577 | 0.052 | 0.000 | 0.073 |
| word-list recall | machine vs mean of A and B | 576 | 0.281 | 0.071 | 0.255 |

## Figure

`study2_agreement_by_measure.png`: ICC(2,1) per measure, machine vs human in teal (mean of the two passes, whisker spanning them) and human vs human in amber, reference line at 0.90.
