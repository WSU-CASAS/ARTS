
# ARTS: Automated Recall Transcription and Scoring

ARTS transcribes and scores spoken word-list recall recorded on a smartwatch. Each
recording is transcribed with OpenAI's open-source Whisper model (medium, used without
fine-tuning, prompts or vocabulary lists), and the transcript is scored with
deterministic, versioned rules for correct recall, repetitions, intrusions, and serial
position (primacy, middle, recency). Every decision is recorded per word, so each score
can be traced to the rule that produced it.

This repository contains the recall pipeline evaluated in the paper *Automated Scoring of
Verbal Recall in Smartwatch-Based Ecological Momentary Assessment: Development and
Independent Validation* (submitted). It was developed at Washington State University by
the CASAS group and the Neuropsychology and Aging Laboratory.

## How it works

```
recording (.m4a) --> Whisper transcript --> base scoring rules --> refinements --> scores
                     (+ word probabilities                         (R2 to R10)     (+ per-word audit table)
                        and alternatives)
```

- **Base scoring rules**: each word of Whisper's transcript is matched to the list studied
  in that session (and to earlier lists); a repeated word is a repetition, an off-list word
  an intrusion, and fillers and comments ("I forgot the rest") are set aside.
- **Refinements** handle known transcription errors: candidate rescue (a list word among
  Whisper's own alternatives for a word it was unsure of), a near-miss rule for the first
  three words, compound splitting, removal of a trailing low-confidence "you", retractions,
  self-corrections, and counting intrusions only within the recall itself.
- **Two configurations**: `automatic` uses no human input; `ra_informed` (the default and
  the final pipeline in the paper) also applies a small confusion map of word pairs that
  research assistants verified by listening (for example "Jim" heard for *gym*), used only
  when the intended word is on that session's list.

The full rule list, with the reason for every threshold, is in [docs/METHODS.md](docs/METHODS.md)
and next to each constant in [src/recipe.py](src/recipe.py).

## Recipe versions

| Version | What it is |
|---|---|
| `2026-09-freeze` (default) | The frozen recipe evaluated in the paper. Produces the published numbers. |
| `2026-09-v2` (exploratory) | Adds a guard against Whisper's repeated-segment decoding loops and a rule for participant narration during the recall. Built after the validation run, so it is not a blind result; see [docs/results/study2_v2_exploratory.md](docs/results/study2_v2_exploratory.md). Select it with `SWC_RECIPE_VERSION=2026-09-v2`. |

For new data, the `2026-09-v2` loop guard removes most of the repetitions that Whisper
adds when it writes the same stretch of speech twice.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # Python 3.11 or later; Whisper also needs ffmpeg
make test                              # synthetic transcripts only, no data needed
```

Running on recordings needs the study's audio, the watch export with the word lists, and
(for agreement statistics) manual scores. None of these are in the repository; see
[docs/DATA.md](docs/DATA.md) for the expected layout and [docs/REPRODUCE.md](docs/REPRODUCE.md)
for the full sequence:

```bash
make index                      # trial index and word lists from the watch export
make transcribe MODEL=medium    # Whisper transcripts (cached)
make transcribe-words MODEL=medium
make score                      # scores_word_recall.csv, scores_serial.csv, per-word audit table
make validate                   # development-sample report (needs manual scores)
make study2 PASS_A=... PASS_B=...   # validation-sample agreement with two raters
```

## Results

Agreement with research assistants, ICC(2,1):

| | Correct words | Primacy / middle / recency | Repetitions | Intrusions |
|---|---|---|---|---|
| Development sample, final configuration (602 recordings, 14 participants) | .983 | .970 / .970 / .976 | .592 | .711 |
| Validation sample, pipeline vs Rater 1 / Rater 2 (585 recordings from 20 participants; 583 rated by Rater 2) | .971 / .968 | .950 to .974 | .079 / .086 | .175 / .192 |
| Validation sample, Rater 1 vs Rater 2 | .987 | .970 to .990 | .781 | .891 |

The aggregate reports are in [docs/results/](docs/results/). In those files and in the
code, pass A is Rater 1 and pass B is Rater 2.

## Repository layout

```
config.py                  paths and settings (all data live outside the repository)
src/                       indexing, transcription, the frozen recipe and the scorers
tools/pipeline/            alternatives pass, validation report, Study 2 agreement, driver
tools/analysis/            scripts behind two supplement results (transcript drift, candidate rescue)
tests/                     synthetic-transcript tests
docs/                      methods, data layout, reproduction steps, aggregate results
keys/                      where the study's word lists go (not tracked; see keys/README.md)
```

## Data and privacy

The repository holds code and aggregate results only. It contains no recordings,
transcripts, per-recording scores, word lists, or participant identifiers; `cache/`,
`outputs/`, and the files in `keys/` are gitignored.

## License

MIT; see [LICENSE](LICENSE).
