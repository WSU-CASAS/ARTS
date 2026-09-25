"""Shared helpers: filename parsing, audio file indexing, text normalization, WER."""
from __future__ import annotations
import _bootstrap  # noqa: F401  (path shim, must be first)
import io
import re
import os
import glob
import zipfile
from functools import lru_cache

import pandas as pd

import config

# --- Filename parsing -------------------------------------------------------
# e.g. recording_catwsu000_memory_recall_20250101_090000-0700.m4a (made-up example)
_FNAME_RE = re.compile(
    r"recording_(?P<device>cat[a-z]+\d+[a-z]?)_(?P<task>.+?)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})(?P<tz>[-+]\d{4})?\.m4a$"
)

TASK_CANON = {
    "memory_recall": "word_recall",
    "verbal_fluency_letter": "letter_fluency",
    "verbal_fluency_category": "category_fluency",
    "voice_journal": "voice_journal",
}


def parse_basename(name: str) -> dict | None:
    """Return {device, participant_num, task, date, time, tz} or None."""
    m = _FNAME_RE.search(str(name))
    if not m:
        return None
    d = m.groupdict()
    num = re.search(r"(\d+)", d["device"])
    d["participant_num"] = int(num.group(1)) if num else None
    d["task_canon"] = TASK_CANON.get(d["task"], d["task"])
    d["basename"] = os.path.basename(str(name))
    return d


# --- Audio file indexing (dedup by basename) --------------------------------
def index_audio_files() -> pd.DataFrame:
    """One row per *unique* basename, with a canonical resolved path.

    The dataset stores the same recording in both the cumulative `catwsu/`
    folder and dated snapshot folders (up to 4 copies). We dedup by basename
    and prefer the copy under CANONICAL_TOP_FOLDER.
    """
    paths = glob.glob(str(config.AUDIO_ROOT / "**" / "*.m4a"), recursive=True)
    rows = []
    for p in paths:
        rel = os.path.relpath(p, config.AUDIO_ROOT)
        top = rel.split(os.sep)[0]
        rows.append({"path": p, "basename": os.path.basename(p), "top_folder": top})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["is_canonical_top"] = (df["top_folder"] == config.CANONICAL_TOP_FOLDER).astype(int)
    df = df.sort_values(["basename", "is_canonical_top"], ascending=[True, False])
    df = df.drop_duplicates("basename", keep="first").reset_index(drop=True)
    return df[["basename", "path", "top_folder"]]


# --- Reading xlsx members straight out of the prompts zip -------------------
@lru_cache(maxsize=None)
def read_zip_xlsx(member: str, header=0) -> pd.DataFrame:
    with zipfile.ZipFile(config.PROMPTS_ZIP) as z:
        with z.open(member) as fh:
            data = fh.read()
    return pd.read_excel(io.BytesIO(data), header=header, engine="openpyxl")


# --- Text normalization & token extraction ----------------------------------
_FILLERS = {
    "um", "uh", "uhm", "er", "ah", "okay", "ok", "and", "the", "a", "an",
    "so", "like", "hmm", "well", "let", "me", "see", "i", "think",
}
_PUNCT_RE = re.compile(r"[^a-z']+")


def normalize_word(w: str) -> str:
    w = str(w).strip().lower()
    w = _PUNCT_RE.sub("", w)
    return w


def tokens_from_text(text: str, drop_fillers: bool = True) -> list[str]:
    """Split a free-recall / fluency transcript into candidate word tokens."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    raw = re.split(r"[,\n;]+|\s+", str(text).lower())
    out = []
    for t in raw:
        t = normalize_word(t)
        if not t:
            continue
        if drop_fillers and t in _FILLERS:
            continue
        out.append(t)
    return out


def singular(w: str) -> str:
    """Very light de-pluralization for lenient matching (cars->car, boxes->box).

    The -es rule applies only to sibilant stems (dishes->dish, boxes->box);
    e-final nouns fall through to plain -s stripping (cakes->cake, stones->stone).
    """
    if len(w) > 3 and re.search(r"(?:sh|ch|ss|x|z)es$", w):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def word_matches(candidate: str, target: str, lenient: bool = True) -> bool:
    c, t = normalize_word(candidate), normalize_word(target)
    if c == t:
        return True
    if lenient and singular(c) == singular(t):
        return True
    return False


# --- Word Error Rate --------------------------------------------------------
def _levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(reference: str, hypothesis: str) -> tuple[float, int]:
    """Word error rate (edit distance / #ref words). Returns (wer, n_ref)."""
    ref = tokens_from_text(reference, drop_fillers=False)
    hyp = tokens_from_text(hypothesis, drop_fillers=False)
    if not ref:
        return (float("nan"), 0)
    return (_levenshtein(ref, hyp) / len(ref), len(ref))
