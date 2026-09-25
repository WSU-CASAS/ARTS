"""Validate the automated pipeline against human references.

The human transcription workbook (config.HUMAN_XLSX, tab "Memory Recall") holds
verified manual transcripts. This script runs in two modes:

  WER             Whisper text vs human transcript -> word error rate (per task).

  Score agreement automated score (on the Whisper transcript = the real pipeline
                  output) vs a REFERENCE score, with Pearson / Spearman / ICC(2,1)
                  / MAE / bias / Bland-Altman limits / exact & within-1 rates.

                  Reference is chosen automatically:
                   - a human numeric score column if one exists (real validation), or
                   - else the score computed on the human *transcript* ("gold"
                     proxy), so the agreement math runs and is verified TODAY.

Point --human-score-col at a numeric score column to use real manual scores
(auto-detected when present).

    python src/validate.py
    python src/validate.py --reference gold          # force the transcript proxy
    python src/validate.py --human-score-col Score   # once real scores exist
"""
from __future__ import annotations
import _bootstrap  # noqa: F401
import argparse
import numpy as np
import pandas as pd

import config
import common
from transcribe import load_transcript
from score_wordrecall import score_recall

SHEET_TASK = {
    "Memory Recall": "word_recall",
}
META_COLS = {"ID", "Date", "Time", "Type", "Audio Transcription", "Notes"}


def load_human() -> tuple[pd.DataFrame, list[str]]:
    frames, score_cols = [], set()
    for sheet, task in SHEET_TASK.items():
        df = pd.read_excel(config.HUMAN_XLSX, sheet_name=sheet, header=1, engine="openpyxl")
        df = df.rename(columns={c: str(c).strip() for c in df.columns})
        # any non-meta column carrying data is a candidate human score column
        for c in df.columns:
            if c not in META_COLS and not c.startswith("Unnamed") and df[c].notna().any():
                score_cols.add(c)
        frames.append(df.assign(task=task))
    h = pd.concat(frames, ignore_index=True).dropna(subset=["ID", "Date", "Time"])
    h["participant_num"] = h["ID"].astype(str).str.extract(r"(\d+)").astype("Int64")
    h["dkey"] = h["Date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    # zero-pad so a time typed as 9:32:33 matches the recording's 093233
    h["tkey"] = h["Time"].astype(str).str.replace(r"\D", "", regex=True).str[:6].str.zfill(6)
    h["join_key"] = h["participant_num"].astype(str) + "_" + h["dkey"] + "_" + h["tkey"]
    return h, sorted(score_cols)


def index_join_keys() -> pd.DataFrame:
    idx = pd.read_csv(config.INDEX_CSV).dropna(subset=["participant_num", "date", "time"])
    idx["join_key"] = (idx["participant_num"].astype("Int64").astype(str) + "_"
                       + idx["date"].astype("Int64").astype(str).str.zfill(8) + "_"
                       + idx["time"].astype("Int64").astype(str).str.zfill(6))
    return idx[["basename", "join_key", "task", "stimulus"]]


def primary_score(task: str, text: str, stim):
    """The headline automated count (correct words), given any transcript."""
    if text is None or not isinstance(stim, str) or not stim:
        return np.nan
    if task == "word_recall":
        return score_recall(text, [t for t in stim.split("|") if t])["n_correct"]
    return np.nan


# --- agreement statistics ---------------------------------------------------
def _spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def _icc21(human, auto):
    """ICC(2,1): two-way random effects, single rater, absolute agreement."""
    X = np.column_stack([human, auto]).astype(float)
    n, k = X.shape
    if n < 2:
        return np.nan
    grand = X.mean()
    SSR = k * ((X.mean(1) - grand) ** 2).sum()        # between subjects
    SSC = n * ((X.mean(0) - grand) ** 2).sum()         # between raters
    SSE = ((X - grand) ** 2).sum() - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) else np.nan
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return float((MSR - MSE) / denom) if denom else np.nan


def agreement_stats(human, auto) -> dict:
    h, a = np.asarray(human, float), np.asarray(auto, float)
    m = ~(np.isnan(h) | np.isnan(a))
    h, a = h[m], a[m]
    n = len(h)
    if n < 2:
        return {"n": n}
    diff = a - h
    pearson = float(np.corrcoef(h, a)[0, 1]) if np.std(h) and np.std(a) else np.nan
    return {
        "n": n,
        "pearson_r": round(pearson, 3),
        "spearman_rho": round(_spearman(h, a), 3),
        "icc21": round(_icc21(h, a), 3),
        "mae": round(float(np.mean(np.abs(diff))), 2),
        "bias_auto_minus_ref": round(float(np.mean(diff)), 2),
        "loa_lower": round(float(np.mean(diff) - 1.96 * np.std(diff)), 2),
        "loa_upper": round(float(np.mean(diff) + 1.96 * np.std(diff)), 2),
        "exact_match_rate": round(float(np.mean(diff == 0)), 2),
        "within1_rate": round(float(np.mean(np.abs(diff) <= 1)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-score-col", default=None,
                    help="name of the human numeric score column (auto-detected if omitted)")
    ap.add_argument("--reference", default="auto", choices=["auto", "gold", "human"],
                    help="auto: human col if present else gold-transcript proxy")
    args = ap.parse_args()

    human, detected = load_human()
    idx = index_join_keys()
    merged = human.merge(idx, on="join_key", how="left", suffixes=("_h", "_i"))
    matched = merged.dropna(subset=["basename"]).copy()
    print(f"[validate] human rows: {len(human)} | matched to an audio trial: "
          f"{len(matched)} ({len(matched)/len(human):.0%})")

    # choose reference
    score_col = args.human_score_col or (detected[0] if detected else None)
    if args.reference == "human" and not score_col:
        print("[validate] --reference human but no human score column found; aborting.")
        return
    use_human = (args.reference in ("auto", "human")) and bool(score_col)
    ref_name = f"human:{score_col}" if use_human else "gold-transcript proxy"
    if detected:
        print(f"[validate] human score column(s) detected: {detected}")
    else:
        print("[validate] no human numeric scores yet -> using gold-transcript proxy.")
    print(f"[validate] agreement reference = {ref_name}")

    rows = []
    for _, r in matched.iterrows():
        whisper = load_transcript(r["basename"])
        if whisper is None:
            continue
        gold = r["Audio Transcription"]
        w, n_ref = common.wer(gold, whisper)
        auto = primary_score(r["task_h"], whisper, r["stimulus"])
        gold_s = primary_score(r["task_h"], gold, r["stimulus"])
        human_s = (pd.to_numeric(r.get(score_col), errors="coerce")
                   if use_human and score_col in matched.columns else np.nan)
        ref = human_s if use_human else gold_s
        rows.append({"task": r["task_h"], "basename": r["basename"], "wer": w,
                     "n_ref_words": n_ref, "auto_score": auto, "gold_score": gold_s,
                     "human_score": human_s, "reference_score": ref})

    paired = pd.DataFrame(rows)
    paired.to_csv(config.OUT_DIR / "validation_paired.csv", index=False)

    if len(paired):
        print("\n[WER] Whisper vs human transcript, by task:")
        print(paired.groupby("task")["wer"].agg(["count", "mean", "median"]).round(3))

        print(f"\n[agreement] automated (Whisper) vs {ref_name}, by task:")
        usable = paired.dropna(subset=["auto_score", "reference_score"])
        stat_rows = []
        for task, g in usable.groupby("task"):
            s = agreement_stats(g["reference_score"], g["auto_score"])
            s["task"] = task
            stat_rows.append(s)
        if stat_rows:
            cols = ["task", "n", "pearson_r", "spearman_rho", "icc21", "mae",
                    "bias_auto_minus_ref", "loa_lower", "loa_upper",
                    "exact_match_rate", "within1_rate"]
            stats = pd.DataFrame(stat_rows)[cols]
            stats.to_csv(config.OUT_DIR / "validation_agreement.csv", index=False)
            print(stats.to_string(index=False))
        else:
            print("  (no usable paired scores)")
    print("\n[validate] wrote validation_paired.csv & validation_agreement.csv")
    if not use_human:
        print("[note] proxy mode: reference is the score on the HUMAN TRANSCRIPT. "
              "Swap in real human scores with --human-score-col once available.")


if __name__ == "__main__":
    main()
