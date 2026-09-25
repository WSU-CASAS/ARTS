#!/usr/bin/env python3
"""Text-drift check between two Whisper transcript caches (READ-ONLY).

Answers, in order of decisiveness:

  1. Does the word-recall SCORE move?  (score_wordrecall.score_recall re-run on
     both texts; n_correct is the quantity ICC(2,1)=0.971 was computed from.)
  2. If yes, on which trials, and are any of them in the ICC validation set?
  3. Only then: how much raw text drifted (exact-match rate, WER), and whether
     the drift is punctuation/filler noise or a scoreable word.

Writes nothing anywhere unless --report-json is given (and that path must be
outside the repo). Never touches either cache dir.

    python3 tools/pipeline/compare_transcript_caches.py OLD_DIR NEW_DIR \
        --validation-csv outputs/validation_paired.csv
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path


# --------------------------------------------------------------------------
# repo imports (src/ is not a package; mirror _bootstrap's path shim)
# --------------------------------------------------------------------------
def load_repo(repo: Path):
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo))
    import common                      # noqa: E402
    import score_wordrecall as swr     # noqa: E402
    import config                      # noqa: E402
    return common, swr, config


# --------------------------------------------------------------------------
# normalisation ladder
# --------------------------------------------------------------------------
def norm_raw(t: str) -> str:
    """Level 0 - byte-ish identity, only trailing/leading whitespace collapsed."""
    return " ".join(str(t or "").split())


def norm_case_punct(common, t: str) -> str:
    """Level 1 - lowercase, punctuation stripped, whitespace collapsed.
    Uses the pipeline's own tokenizer so 'identical' means identical *to the
    scorer's eyes*, not to a diff tool's."""
    return " ".join(common.tokens_from_text(t, drop_fillers=False))


def norm_content(common, t: str) -> str:
    """Level 2 - level 1 minus fillers (um/uh/okay/and/the/...). Two texts equal
    here but different at level 1 differ ONLY in words the scorer deletes."""
    return " ".join(common.tokens_from_text(t, drop_fillers=True))


def punct_signature(t: str) -> str:
    """Clause-boundary punctuation only. score_wordrecall._clauses splits on
    [.,!?;\\n] and meta-speech patterns are anchored per clause, so punctuation
    drift alone can change a score even when the words are identical."""
    return "".join(c for c in str(t or "") if c in ".,!?;\n")


# --------------------------------------------------------------------------
# cache IO
# --------------------------------------------------------------------------
def read_cache(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception as e:
            out[p.name[:-5]] = {"_error": str(e)}
            continue
        out[p.name[:-5]] = j
    return out


def pct(a, b):
    return (100.0 * a / b) if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_dir")
    ap.add_argument("new_dir")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]),
                    help="repo root (default: the repo this script lives in)")
    ap.add_argument("--index", default=None, help="trial_index.csv (default <repo>/outputs/trial_index.csv)")
    ap.add_argument("--validation-csv", default=None,
                    help="validation_paired.csv - trials the ICC was computed on")
    ap.add_argument("--task", default="word_recall",
                    help="task to re-score ('all' = score nothing, text metrics only)")
    ap.add_argument("--show", type=int, default=25, help="max example diffs to print")
    ap.add_argument("--report-json", default=None, help="optional machine-readable dump (must be outside the repo)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    old_dir, new_dir = Path(args.old_dir).resolve(), Path(args.new_dir).resolve()
    for d in (old_dir, new_dir):
        if not d.is_dir():
            sys.exit(f"not a directory: {d}")
    if old_dir == new_dir:
        sys.exit("old and new cache dirs are the same")
    if args.report_json and repo in Path(args.report_json).resolve().parents:
        sys.exit("refusing to write a report inside the repo (this tool is read-only)")

    common, swr, config = load_repo(repo)
    import pandas as pd

    old, new = read_cache(old_dir), read_cache(new_dir)
    ko, kn = set(old), set(new)
    both = sorted(ko & kn)

    print(f"old cache : {old_dir}  ({len(ko)} files)")
    print(f"new cache : {new_dir}  ({len(kn)} files)")
    print(f"overlap   : {len(both)} | old-only {len(ko - kn)} | new-only {len(kn - ko)}")

    bad = [k for k in both if "_error" in old[k] or "_error" in new[k]]
    if bad:
        print(f"!! {len(bad)} unreadable JSON on one side, e.g. {bad[:3]}")

    # --- model sanity: a model mismatch invalidates the whole comparison -----
    models_old = {json.dumps(old[k].get("model")) for k in both}
    models_new = {json.dumps(new[k].get("model")) for k in both}
    print(f"model(old): {sorted(models_old)}   model(new): {sorted(models_new)}")
    if models_old != models_new:
        print("!! MODEL MISMATCH - drift below is not attributable to word_timestamps.")

    # --- word-onset presence on the new side --------------------------------
    n_words_present = sum(1 for k in both if new[k].get("words"))
    n_empty_text = sum(1 for k in both if not str(new[k].get("text") or "").strip())
    print(f"new entries carrying per-word onsets: {n_words_present}/{len(both)}"
          f"  (empty-text entries: {n_empty_text})")

    # --- text drift ladder ---------------------------------------------------
    rows = []
    for k in both:
        to, tn = old[k].get("text") or "", new[k].get("text") or ""
        r0o, r0n = norm_raw(to), norm_raw(tn)
        r1o, r1n = norm_case_punct(common, to), norm_case_punct(common, tn)
        r2o, r2n = norm_content(common, to), norm_content(common, tn)
        w, nref = common.wer(to, tn)
        rows.append({
            "basename": k, "text_old": to, "text_new": tn,
            "eq_raw": r0o == r0n,
            "eq_norm": r1o == r1n,
            "eq_content": r2o == r2n,
            "eq_punct": punct_signature(to) == punct_signature(tn),
            "wer": w, "n_ref": nref,
            "ntok_old": len(r1o.split()), "ntok_new": len(r1n.split()),
        })
    df = pd.DataFrame(rows)
    n = len(df)

    print("\n--- text drift -------------------------------------------------")
    print(f"L0 exact (raw)                    : {df.eq_raw.sum()}/{n}  ({pct(df.eq_raw.sum(), n):.2f}%)")
    print(f"L1 exact (case/punct/ws normed)   : {df.eq_norm.sum()}/{n}  ({pct(df.eq_norm.sum(), n):.2f}%)")
    print(f"L2 exact (fillers also dropped)   : {df.eq_content.sum()}/{n}  ({pct(df.eq_content.sum(), n):.2f}%)")
    print(f"clause punctuation identical      : {df.eq_punct.sum()}/{n}  ({pct(df.eq_punct.sum(), n):.2f}%)")
    tot_wer = (df.wer.fillna(0) * df.n_ref).sum() / max(df.n_ref.sum(), 1)
    print(f"corpus WER old->new               : {tot_wer:.5f}   "
          f"(per-file mean {df.wer.mean(skipna=True):.5f}, max {df.wer.max(skipna=True):.4f})")
    print(f"token count delta                 : sum {int((df.ntok_new - df.ntok_old).sum()):+d} "
          f"| files with |delta|>0: {int((df.ntok_new != df.ntok_old).sum())}")

    # drift taxonomy on the files that are not L0-identical
    d1 = df[~df.eq_raw]
    only_punct_ws = int((d1.eq_norm).sum())                       # words same, punctuation/case moved
    only_filler = int((~d1.eq_norm & d1.eq_content).sum())        # only filler words moved
    scoreable = int((~d1.eq_content).sum())                       # a NON-filler word moved
    print(f"\nof {len(d1)} non-identical files: punct/case-only {only_punct_ws} | "
          f"filler-only {only_filler} | SCOREABLE-word change {scoreable}")

    # --- the decisive metric: does the score move? --------------------------
    idx_path = Path(args.index) if args.index else repo / "outputs" / "trial_index.csv"
    score_df = None
    if args.task != "all" and idx_path.exists():
        idx = pd.read_csv(idx_path)
        idx = idx[(idx["task"] == args.task) & idx["audio_found"]]
        priors = swr.prior_words_map(idx) if args.task == "word_recall" else {}
        srows = []
        for _, r in idx.iterrows():
            k = r["basename"]
            if k not in old or k not in new:
                continue
            targets = [t for t in str(r["stimulus"]).split("|") if t]
            pw = priors.get(k)
            so = swr.score_recall(old[k].get("text") or "", targets, prior_words=pw)
            sn = swr.score_recall(new[k].get("text") or "", targets, prior_words=pw)
            srows.append({
                "basename": k, "participant_num": r["participant_num"],
                "n_correct_old": so["n_correct"], "n_correct_new": sn["n_correct"],
                "d_correct": sn["n_correct"] - so["n_correct"],
                "d_intrusions": sn["n_intrusions"] - so["n_intrusions"],
                "d_repetitions": sn["n_repetitions"] - so["n_repetitions"],
                "d_deleted": sn["n_deleted"] - so["n_deleted"],
                "text_old": old[k].get("text") or "", "text_new": new[k].get("text") or "",
            })
        score_df = pd.DataFrame(srows)

    if score_df is not None and len(score_df):
        s = score_df
        moved = s[s.d_correct != 0]
        print(f"\n--- SCORE drift ({args.task}, n={len(s)}) ------------------------")
        print(f"n_correct identical               : {len(s) - len(moved)}/{len(s)}  "
              f"({pct(len(s) - len(moved), len(s)):.2f}%)")
        print(f"trials with |delta n_correct| >= 1     : {len(moved)}   "
              f"(sum {int(s.d_correct.sum()):+d}, mean {s.d_correct.mean():+.4f}, "
              f"max|delta| {int(s.d_correct.abs().max())})")
        print(f"mean |delta| n_correct                 : {s.d_correct.abs().mean():.4f}")
        if len(s) > 1 and s.n_correct_old.std() > 0 and s.n_correct_new.std() > 0:
            print(f"pearson r(old, new) n_correct     : {s.n_correct_old.corr(s.n_correct_new):.6f}")
        for c in ("d_intrusions", "d_repetitions", "d_deleted"):
            print(f"{c:34s}: nonzero on {int((s[c] != 0).sum())} trials")

        # ICC subset: the only trials the 0.971 number can move on
        if args.validation_csv:
            vp = Path(args.validation_csv)
            if not vp.is_absolute():
                vp = repo / vp
            if vp.exists():
                v = pd.read_csv(vp)
                vb = set(v[v["task"] == args.task]["basename"]) if "task" in v else set(v["basename"])
                sub = s[s.basename.isin(vb)]
                submoved = sub[sub.d_correct != 0]
                print(f"\nICC validation subset             : {len(sub)} trials "
                      f"({len(vb)} listed in {vp.name})")
                print(f"  score changes inside that subset: {len(submoved)}"
                      f"   sum delta {int(sub.d_correct.sum()):+d}")
                if len(submoved) == 0:
                    print("  => ICC(2,1) is UNCHANGED by construction (same auto scores,"
                          " same reference). No re-validation needed.")
                else:
                    print("  => ICC WILL move. Re-run validate.py against the new cache"
                          " before adopting.")
            else:
                print(f"\n(validation csv not found: {vp})")

        if len(moved):
            print(f"\n--- trials whose score changed (showing {min(args.show, len(moved))}) ---")
            for _, r in moved.head(args.show).iterrows():
                print(f"\n{r.basename}  n_correct {r.n_correct_old} -> {r.n_correct_new} "
                      f"({r.d_correct:+d})")
                print(f"  OLD: {r.text_old[:300]}")
                print(f"  NEW: {r.text_new[:300]}")

    # --- example text diffs that are NOT score-affecting --------------------
    show = df[~df.eq_content].head(args.show)
    if len(show):
        print(f"\n--- content-word text diffs (showing {len(show)}) ---")
        for _, r in show.iterrows():
            print(f"\n{r.basename}  wer={r.wer:.3f}")
            print(f"  OLD: {r.text_old[:300]}")
            print(f"  NEW: {r.text_new[:300]}")

    # --- verdict ------------------------------------------------------------
    print("\n--- verdict ----------------------------------------------------")
    ok_cov = (len(both) == len(ko)) and not (ko - kn)
    ok_model = models_old == models_new
    ok_score = score_df is not None and len(score_df) and (score_df.d_correct != 0).sum() == 0
    ok_wer = tot_wer <= 0.005
    ok_exact = pct(df.eq_norm.sum(), n) >= 99.0
    for label, ok in [("coverage: new cache covers every old basename", ok_cov),
                      ("same whisper model on both sides", ok_model),
                      ("word-recall n_correct identical on every trial", ok_score),
                      ("corpus WER <= 0.005", ok_wer),
                      ("L1 exact-match >= 99.0%", ok_exact)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("\nADOPT WHOLESALE only if all five pass. Any n_correct change means the"
          "\nvalidated ICC must be recomputed before the new cache replaces the old.")

    if args.report_json:
        payload = {
            "old_dir": str(old_dir), "new_dir": str(new_dir),
            "n_overlap": len(both), "old_only": sorted(ko - kn)[:200],
            "new_only": sorted(kn - ko)[:200],
            "corpus_wer": tot_wer,
            "eq_raw": int(df.eq_raw.sum()), "eq_norm": int(df.eq_norm.sum()),
            "eq_content": int(df.eq_content.sum()), "n": n,
            "scoreable_changes": scoreable,
            "score_changed_trials": ([] if score_df is None or not len(score_df) else
                                     score_df[score_df.d_correct != 0]
                                     [["basename", "n_correct_old", "n_correct_new"]]
                                     .to_dict("records")),
        }
        Path(args.report_json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.report_json}")

    changed = 0 if (ok_cov and ok_model and ok_score) else 1
    return changed


if __name__ == "__main__":
    sys.exit(main())
