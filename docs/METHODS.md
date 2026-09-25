
# Methods: the frozen scoring recipe and its validation

This page describes what the code does and how it was validated. Numbers come from the
aggregate reports in `docs/results/`, which the scripts named below regenerate from the
study data.

## 1. Task and recordings

Participants learned a 12-word list that the watch read aloud and showed on screen, then
spoke every word they remembered while the watch recorded; delayed recall followed on
alternate days. Each recording is one `.m4a` file whose name carries the watch code, the
task, and the date and time (`recording_<device>_memory_recall_<YYYYMMDD>_<HHMMSS><tz>.m4a`).
`src/io_index.py` builds the trial index from the watch export and resolves each file.

Scores per recording: correct words, repetitions, intrusions (new and from earlier lists),
and correct words by serial position (primacy 1 to 3, middle 4 to 9, recency 10 to 12).

## 2. Transcription

Every recording is transcribed once with OpenAI Whisper, model `medium`, package
`openai-whisper` 20250625, `language="en"`, `fp16=False`, with no fine-tuning, prompt, or
vocabulary list (`src/transcribe.py`). Whisper never sees the word list. Transcripts are
cached as JSON under `cache/transcripts/` and are not rewritten. A second decode of the same
audio with `word_timestamps=True` goes to `cache/transcripts_words/`; it supplies per-word
probabilities and onsets and is never the transcript that is scored.

A third pass, teacher forcing (`tools/pipeline/word_alternatives.py`), reads each production
transcript back through the model and records, for every word, the probability of the word
written, the entropy of the distribution, and the five most probable alternatives. This
table feeds candidate rescue (R3).

## 3. Scoring rules (recipe version `2026-09-freeze`)

All rules live in `src/recipe.py`, with the reason for every threshold next to the constant.
`src/score_wordrecall.py` records the rule that touched each token in a per-token note
(`tools/pipeline/export_token_classification.py` writes the table), so every credited or
excluded word can be audited.

Base scoring rules (the starting point of both configurations):

- R1 Normalisation: lower case, punctuation stripped, fillers removed, meta-speech clauses
  ("that is all I can remember") removed. Each remaining word is matched to the session's
  list (plural forms accepted) and to earlier lists; a repeated list word is a repetition and
  an off-list word an intrusion.

Refinements:

- R2 List-gated confusion map (research assistant-informed configuration only): word pairs
  that research assistants verified by listening (for example "Jim" for *gym*), applied only
  when the intended word is on that recording's own list.
- R3 Candidate rescue: when Whisper was unsure of a written word that is not on the list
  (probability below 0.50, entropy at most 5.0) and a list word is among its alternatives with
  probability at least 0.05, the list word is credited. The rule adds credit only for list
  words not already credited and never changes the transcript.
- R4 Near-miss rule: a word one letter away from a list word is credited, but only among the
  first three content words.
- R5 Compound split: a merged word of six or more letters that is one edit from two list words
  run together is split and both are credited.
- R6 Trailing artifact: a final "you" with word probability below 0.15, a known Whisper
  end-of-audio artifact, is dropped.
- R7 Retraction: a word followed at once by "no" loses credit unless the "no" opens a give-up
  phrase; "no" itself is never an intrusion.
- R8 Self-correction: a word named inside a self-correction ("did I already say bell") is not
  a repetition.
- R9 Recall span: off-list words count as intrusions only from the first list word through
  the words said right after the last one, up to the first filler or comment.
- R10 A repeated intrusion counts as both a repetition and an intrusion.

Serial position (`src/score_serial.py`) counts correct words by their position in the
presentation order (`keys/word_lists_ordered.json`, supplied by the study).

Configurations and provenance:

- Two configurations are first class: `automatic` (base rules plus R3 to R10, no human input)
  and `ra_informed` (the default: adds R2). Select with `SWC_RECIPE_CONFIG`.
- A manifest (`outputs/recipe_manifest.json`) records the recipe version, configuration,
  every threshold, the git commit, input hashes and the time of each run.

## 4. Statistics

Agreement is the intraclass correlation ICC(2,1): two-way random effects, single rater,
absolute agreement (`validate.agreement_stats`), with mean absolute difference, bias
(machine minus reference), exact-match rate and within-one rate. Transcript agreement is the
word error rate (`common.wer`).

## 5. Study 1: development and freeze

Rules were developed against the 602 recall recordings, from 14 development participants,
that research assistants had scored when development began, and kept only when agreement
improved. Correct recall: ICC .971 with the base rules, .980 with the automatic
configuration, .983 with the research assistant-informed configuration (mean absolute
difference 0.27 to 0.17 words). Serial position (268 recordings with a known presentation
order): .970, .970, and .976. Intrusions: .273 counting every off-list word, .711 with R9 and
R10. Repetitions: .592.

Transcription accuracy and word-confidence calibration use 148 verified manual transcripts
(1,466 words; 87.4% of words agree), computed by `tools/pipeline/confidence_calibration.py`.

`tools/pipeline/validate_recipe.py` regenerates the report; with `--check` it recomputes
every number and fails if any moves by more than 0.002.

## 6. Study 2: independent validation

Twenty participants were chosen at random and excluded from every step of rule development;
their study numbers are kept outside the repository (`keys/locked_participants.txt`). Each of their
recordings was rated twice (Rater 1 and Rater 2, called pass A and pass B in the code): a
research assistant transcribed the recording without the word list, and the recording was
then scored from that transcript. The frozen recipe was applied once
(`tools/pipeline/run_final.py`, `tools/pipeline/study2_reliability.py`).

| Measure | Pipeline vs Rater 1 | Pipeline vs Rater 2 | Rater 1 vs Rater 2 |
|---|---|---|---|
| Correct words | .971 | .968 | .987 |
| Primacy / middle / recency | .963 / .954 / .974 | .968 / .950 / .973 | .984 / .970 / .990 |
| Repetitions | .079 | .086 | .781 |
| Intrusions | .175 | .192 | .891 |

Low agreement for repetitions and intrusions comes mainly from their low base rate, from
stretches that Whisper wrote twice, and from commentary or other voices in the recording; the
paper discusses each source.

## 7. Exploratory revision `2026-09-v2`

After the validation run, two failure modes were addressed behind a version switch, with
every threshold chosen on the development recordings: a loop guard for Whisper's
repeated-segment decoding loops and a narration-clause rule for intrusions. The frozen
default is unchanged. Results and the disclosure of what was informed by the validation
sample are in `docs/results/study2_v2_exploratory.md`.
