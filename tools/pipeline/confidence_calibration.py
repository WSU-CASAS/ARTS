#!/usr/bin/env python3
"""Calibration of Whisper's word confidence against verified manual transcripts (recall only).

Each manual transcript in the human transcription workbook (config.HUMAN_XLSX, tab
"Memory Recall") is matched to its recording by participant, date and time; the words of
the word-timestamp decode (cache/transcripts_words/) are aligned to the manual words by
edit distance, and each decoded word is marked as matching or not. Per confidence bin the
script reports the number of words and the share that match.

Writes outputs/word_confidence_calibration.csv (bin, words, accuracy in percent), which
validate_recipe.py copies into the validation report. Aggregates only.

    python3 tools/pipeline/confidence_calibration.py
"""
from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

import pandas as pd

import config
from validate import index_join_keys, load_human

BINS = [0, .2, .4, .6, .8, .95, 1.0001]
LABELS = ["below 0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-0.95", "0.95 and above"]
_norm_re = re.compile(r"[^a-z']+")


def norm_tok(w: str) -> str:
    return _norm_re.sub("", (w or "").lower())


def align_labels(ref: list[str], hyp: list[str]) -> list[bool]:
    """Levenshtein backtrace; one True/False per hypothesis token (True = matches the reference)."""
    R, Hn = len(ref), len(hyp)
    D = [[0] * (Hn + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        D[i][0] = i
    for j in range(Hn + 1):
        D[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, Hn + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    lab = [False] * Hn
    i, j = R, Hn
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            lab[j - 1] = ref[i - 1] == hyp[j - 1]
            i, j = i - 1, j - 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            lab[j - 1] = False          # insertion: Whisper wrote a word the transcriber did not hear
            j -= 1
        else:
            i -= 1                      # deletion: the transcriber heard a word Whisper missed
    return lab


def calibrate() -> tuple[pd.DataFrame, int, int]:
    h, _ = load_human()
    h = h[h["task"] == "word_recall"]
    m = h.merge(index_join_keys(), on="join_key", how="inner", suffixes=("_h", ""))
    m = m[m["Audio Transcription"].notna()].drop_duplicates("basename")
    rows, used = [], 0
    for _, r in m.iterrows():
        p = config.WORDS_CACHE / (r["basename"] + ".json")
        if not p.exists():
            continue
        words = [w for w in json.loads(p.read_text()).get("words", []) if norm_tok(w.get("word"))]
        ref = [t for t in (norm_tok(x) for x in str(r["Audio Transcription"]).split()) if t]
        if not words or not ref:
            continue
        for w, ok in zip(words, align_labels(ref, [norm_tok(w["word"]) for w in words])):
            if w.get("probability") is not None:
                rows.append({"prob": float(w["probability"]), "correct": bool(ok)})
        used += 1
    cal = pd.DataFrame(rows)
    cal["bin"] = pd.cut(cal["prob"], bins=BINS, labels=LABELS, right=False)
    ct = (cal.groupby("bin", observed=True).agg(words=("correct", "size"), accuracy=("correct", "mean"))
          .reset_index())
    ct["accuracy"] = (ct["accuracy"] * 100).round(1)
    return ct, used, len(cal)


def main() -> int:
    ct, used, n = calibrate()
    out = config.OUT_DIR / "word_confidence_calibration.csv"
    ct.to_csv(out, index=False)
    overall = (ct["words"] * ct["accuracy"]).sum() / ct["words"].sum()
    print(f"[calibration] {used} recall recordings, {n:,} words, overall {overall:.1f}% -> {out}")
    print(ct.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
