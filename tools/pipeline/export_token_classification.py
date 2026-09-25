"""Export per-token classifications for word-list recall, in two formats
(Maureen's taxonomy: correct / repetition / intrusion-prior-list /
intrusion-new / deleted, with nothing silently dropped).

WIDE  outputs/recall_tokens_wide.csv: one row per recording: summary counts
      + readable `token_classification` cell + never-recalled targets + transcript.
LONG  outputs/recall_tokens_long.csv: one row per word event (plus one row
      per never-recalled target), for filtering/pivoting item-level analyses.

Both derive from the same classify_tokens() the validated scorer uses (frozen
recipe, with the alternatives table and word-level cache looked up per
recording), so they always agree with the official counts. The long table
carries in_span (recipe R9): intrusion tokens outside the recall span keep
their label and the note "outside recall span" but are not counted.

    python tools/pipeline/export_token_classification.py
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

import pandas as pd

import config
from transcribe import load_transcript
from score_wordrecall import classify_tokens, is_repeated_intrusion, prior_words_map

SHORT = {"correct": "correct", "repetition": "repetition",
         "intrusion-prior-list": "intrusion-prior", "intrusion-new": "intrusion-new",
         "deleted": "deleted"}


def load_rows():
    """Yield (meta, tokens, never, transcript) per scored recording."""
    idx = pd.read_csv(config.INDEX_CSV)
    idx = idx[(idx["task"] == "word_recall") & idx["audio_found"]]
    priors = prior_words_map(idx)
    for _, r in idx.iterrows():
        tr = load_transcript(r["basename"])
        if tr is None:
            continue
        targets = [t for t in str(r["stimulus"]).split("|") if t]
        tokens, never = classify_tokens(tr, targets, prior_words=priors.get(r["basename"]),
                                        basename=r["basename"])
        d = str(int(r["date"])) if pd.notna(r["date"]) else ""
        t = str(int(r["time"])).zfill(6) if pd.notna(r["time"]) else ""
        meta = {
            "basename": r["basename"],
            "participant": f"CAT {int(r['participant_num']):03d}" if pd.notna(r["participant_num"]) else "",
            "date": f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d,
            "time": f"{t[0:2]}:{t[2:4]}:{t[4:6]}" if len(t) == 6 else t,
            "list_id": r["list_id"],
            "n_targets": len(targets),
        }
        yield meta, tokens, never, tr


def main():
    wide_rows, long_rows = [], []
    for meta, tokens, never, tr in load_rows():
        for t in tokens:
            long_rows.append({**meta, "position": t["position"],
                              "word_said": t["word_said"],
                              "classification": SHORT[t["classification"]],
                              "matched_list_word": t["matched_list_word"],
                              "in_span": t.get("in_span", ""),
                              "note": t["note"]})
        for w in never:
            long_rows.append({**meta, "position": "", "word_said": "",
                              "classification": "never_recalled",
                              "matched_list_word": w, "in_span": "", "note": ""})

        def cell(t):
            base = f"{t['word_said']}={SHORT[t['classification']]}"
            if t["matched_list_word"] and t["matched_list_word"] != t["word_said"]:
                base += f"({t['matched_list_word']})"
            if t["classification"] == "deleted":
                base += f"[{t['note']}]"
            return base
        n = lambda lab: sum(1 for t in tokens if t["classification"] == lab)
        n_in = lambda lab: sum(1 for t in tokens if t["classification"] == lab and t.get("in_span"))
        n_rep_intr = sum(1 for t in tokens if is_repeated_intrusion(t) and t.get("in_span"))
        wide_rows.append({**meta,
            "n_correct": n("correct"),
            "n_intrusions": n_in("intrusion-prior-list") + n_in("intrusion-new") + n_rep_intr,
            "n_intrusions_prior": n_in("intrusion-prior-list"),
            "n_intrusions_new": n_in("intrusion-new"),
            "n_intrusions_all": n("intrusion-prior-list") + n("intrusion-new"),
            "n_repetitions": n("repetition"),
            "n_deleted": n("deleted"),
            "token_classification": ", ".join(cell(t) for t in tokens),
            "never_recalled_targets": ", ".join(never),
            "whisper_transcript": tr,
        })

    wide = pd.DataFrame(wide_rows).sort_values(["participant", "date", "time"])
    long = pd.DataFrame(long_rows).sort_values(["participant", "date", "time", "position"])
    wide_dst = config.OUT_DIR / "recall_tokens_wide.csv"
    long_dst = config.OUT_DIR / "recall_tokens_long.csv"
    wide.to_csv(wide_dst, index=False)
    long.to_csv(long_dst, index=False)

    # consistency check against the official score file
    score_path = config.OUT_DIR / "scores_word_recall.csv"
    if score_path.exists():
        sc = pd.read_csv(score_path)[["basename", "n_correct", "n_intrusions",
                                      "n_intrusions_prior", "n_intrusions_new",
                                      "n_repetitions", "n_deleted"]]
        m = wide.merge(sc, on="basename", suffixes=("_tok", "_off"))
        bad = m[(m.n_correct_tok != m.n_correct_off) |
                (m.n_intrusions_tok != m.n_intrusions_off) |
                (m.n_intrusions_prior_tok != m.n_intrusions_prior_off) |
                (m.n_intrusions_new_tok != m.n_intrusions_new_off) |
                (m.n_repetitions_tok != m.n_repetitions_off) |
                (m.n_deleted_tok != m.n_deleted_off)]
        print(f"[tokens] consistency vs official scores: {len(m)-len(bad)}/{len(m)} identical"
              + ("" if bad.empty else f"  !! {len(bad)} MISMATCHES"))
    print(f"[tokens] wide: {len(wide)} recordings -> {wide_dst}")
    print(f"[tokens] long: {len(long)} rows -> {long_dst}")


if __name__ == "__main__":
    main()
