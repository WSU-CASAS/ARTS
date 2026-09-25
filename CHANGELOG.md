# Changelog

## 1.0.0 (2026-09-25)

First public release: the recall pipeline behind the paper (recipe `2026-09-freeze`, the
default), with the exploratory recipe `2026-09-v2` behind `SWC_RECIPE_VERSION`.

Changes from the development code, none of which alter any recall score:

- Recall only. The fluency and story scorers of the wider project are not included.
- The held-out participants are read from `keys/locked_participants.txt` (or
  `SWC_LOCKED_PARTICIPANTS`) instead of being listed in the code.
- `validate.load_human` zero-pads the time field, so a manual transcript whose time was typed
  without a leading zero (9:32:33) is matched to its recording (148 verified transcripts, not 147).
- `tools/pipeline/confidence_calibration.py` computes the recall-only calibration used in the
  paper's Supplement S1.
- Whisper defaults to the `medium` model, the one used in the paper.
