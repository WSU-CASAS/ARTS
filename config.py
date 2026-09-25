"""Central paths and settings.

All data live outside the repository (by default one directory above it), so the
repository never holds recordings, transcripts or scores. Override any path with the
environment variable named next to it.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("SWC_PROJECT_ROOT", REPO_ROOT.parent))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, '#' comments.
    Does not overwrite vars already set in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(REPO_ROOT / ".env")

# --- Inputs (read-only) ---
AUDIO_ROOT = Path(os.environ.get("SWC_AUDIO_ROOT", PROJECT_ROOT / "CAT Smart Watch Audios"))
PROMPTS_ZIP = Path(os.environ.get("SWC_PROMPTS_ZIP", PROJECT_ROOT / "CAT Smart Watch Prompts" / "cat_responses_20260629.zip"))
HUMAN_XLSX = Path(os.environ.get("SWC_HUMAN_XLSX", AUDIO_ROOT / "Temporary Audio Transcriptions.xlsx"))

# Spreadsheet member name inside PROMPTS_ZIP (the memory-task export)
MEM_SHEET = os.environ.get("SWC_MEM_SHEET", "cat_mem_20260629.xlsx")

# --- Outputs (generated) ---
KEYS_DIR = REPO_ROOT / "keys"
OUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = REPO_ROOT / "cache"
# Overridable so a word-timestamp pass can write to a FRESH cache without
# disturbing the validated `transcripts/` set (ICC 0.971 is pinned to it).
_tc = Path(os.environ.get("SWC_TRANSCRIPT_CACHE", CACHE_DIR / "transcripts"))
# Resolve a relative override against the repo, not the CWD, so running from
# another directory cannot silently create a stray empty cache.
TRANSCRIPT_CACHE = _tc if _tc.is_absolute() else (REPO_ROOT / _tc)

for _d in (KEYS_DIR, OUT_DIR, TRANSCRIPT_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

INDEX_CSV = OUT_DIR / "trial_index.csv"
WORD_LISTS_JSON = KEYS_DIR / "word_lists.json"
# Presentation order of each list (serial position); supplied by the study, see keys/README.md.
WORD_LISTS_ORDERED_JSON = KEYS_DIR / "word_lists_ordered.json"

# --- Frozen recipe (2026-09-freeze) ---
# Two first-class configurations: "ra_informed" (default) applies the curated
# confusion map (R2); "automatic" disables R2 and nothing else.
RECIPE_CONFIG = os.environ.get("SWC_RECIPE_CONFIG", "ra_informed")
if RECIPE_CONFIG not in ("automatic", "ra_informed"):
    raise ValueError(f"SWC_RECIPE_CONFIG must be 'automatic' or 'ra_informed', got {RECIPE_CONFIG!r}")
# Research-assistant hand scores (read-only, outside the repo).
HAND_MEM_CSV = Path(os.environ.get("SWC_HAND_MEM_CSV",
                                   PROJECT_ROOT / "cat_mem_20250903(cat_mem_20250903).csv"))
# The frozen recipe module (src/recipe.py) is shared with the fluency scorers of the wider
# project and hashes these inputs into its manifest when they exist. The recall pipeline does
# not read them; they are defined so that recipe.py runs unchanged.
FLUENCY_CUES_JSON = KEYS_DIR / "fluency_cues.json"
HAND_LFCF_CSV = Path(os.environ.get("SWC_HAND_LFCF_CSV", PROJECT_ROOT / "hand_scores_fluency.csv"))
PROPER_NOUNS_CSV = KEYS_DIR / "proper_nouns.csv"
# Teacher-forced alternatives (R3), produced by tools/pipeline/word_alternatives.py.
ALTERNATIVES_CSVS = [OUT_DIR / "word_alternatives.csv", OUT_DIR / "word_alternatives_rescue.csv"]
# Word-level Whisper pass (R6 trailing artifact, timing features).
WORDS_CACHE = CACHE_DIR / "transcripts_words"

# --- Transcription ---
WHISPER_MODEL = os.environ.get("SWC_WHISPER_MODEL", "medium")  # the model used in the paper
WHISPER_LANGUAGE = os.environ.get("SWC_WHISPER_LANGUAGE", "en")
# Per-word onsets. Off by default: enabling it invalidates nothing, but the
# timing feature family cannot be computed without it.
WHISPER_WORD_TIMESTAMPS = os.environ.get("SWC_WORD_TIMESTAMPS", "0").lower() \
    not in ("0", "", "false", "no")

# When picking a canonical file among duplicate basenames, prefer a copy whose
# top-level folder under AUDIO_ROOT is this (the cumulative export).
CANONICAL_TOP_FOLDER = "catwsu"

# --- Held-out participants (independent validation) ---
# Study numbers of the participants set aside for validation. They are not stored in the
# repository: list them in keys/locked_participants.txt (one per line or comma separated,
# gitignored) or pass SWC_LOCKED_PARTICIPANTS="8,39,...". Empty when neither is given.
LOCKED_PARTICIPANTS_TXT = KEYS_DIR / "locked_participants.txt"


def _load_locked() -> frozenset:
    raw = os.environ.get("SWC_LOCKED_PARTICIPANTS")
    if raw is None and LOCKED_PARTICIPANTS_TXT.exists():
        raw = LOCKED_PARTICIPANTS_TXT.read_text()
    nums = []
    for line in (raw or "").splitlines():
        line = line.split("#", 1)[0]
        nums += [int(n) for n in re.findall(r"\d+", line)]
    return frozenset(nums)


LOCKED_PARTICIPANTS = _load_locked()
