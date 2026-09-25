#!/usr/bin/env python3
"""Validate candidate-rescue scoring against the full hand-scored recall set.

The rule (as piloted):
  a candidate from alt2/alt3 is credited only if
    - the written word is NOT on the studied list (nothing already credited),
    - the model was unsure of the written word (chosen_prob < 0.5),
    - the candidate IS on the studied list,
    - the candidate clears a probability floor,
    - that list word was never credited elsewhere in the recording,
    - and (entropy gate) the word's entropy is below the flat-distribution
      cutoff, above which the candidate list is noise rather than ambiguity.

Reports before/after agreement with RA hand scores across probability floors,
plus the entropy-gate sensitivity. Writes outputs/rescue_validation.csv.
"""
from __future__ import annotations
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

import pandas as pd

import common
import config
from validate import agreement_stats

PAT = re.compile(r"^(.*)\s+\(([\d.]+)\)$")


def parse(s):
    m = PAT.match(str(s))
    return (common.normalize_word(m.group(1)), float(m.group(2))) if m else (None, 0.0)


def main():
    frames = []
    for f in ["word_alternatives.csv", "word_alternatives_rescue.csv"]:
        p = config.OUT_DIR / f
        if p.exists():
            frames.append(pd.read_csv(p))
    alt = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["basename", "segment_start", "word"])
    alt = alt[alt.basename.str.contains("memory_recall")]

    idx = pd.read_csv(config.INDEX_CSV)
    stim = dict(zip(idx.basename, idx.stimulus))
    auto = pd.read_csv(config.OUT_DIR / "scores_word_recall.csv").set_index("basename")
    tok = pd.read_csv(config.OUT_DIR / "recall_tokens_long.csv")
    nev = (tok[tok.classification == "never_recalled"]
           .groupby("basename")["matched_list_word"].apply(set).to_dict())

    hs = pd.read_csv(config.HAND_MEM_CSV, header=1, dtype=str)
    hs.columns = [c.strip() for c in hs.columns]
    hs = hs[(hs["Prompt Name"] == "Recall") & hs["Recall Audio File Name"].notna()].copy()
    hs["basename"] = hs["Recall Audio File Name"].str.replace(" ", "", regex=False)
    hs["h"] = pd.to_numeric(hs["Correct"], errors="coerce")
    hs = hs.dropna(subset=["h"]).drop_duplicates("basename")
    hmap = dict(zip(hs.basename, hs.h))

    covered = [b for b in hmap if b in set(alt.basename) and b in auto.index]
    print(f"hand-scored recall recordings with alternatives: {len(covered)}")

    rows = []
    for PMIN in (0.02, 0.05, 0.10):
        for HMAX in (3.0, 5.0, 99.0):
            base_a, resc_a, hum = [], [], []
            n_changed = n_rescued_words = 0
            for b in covered:
                g = alt[alt.basename == b]
                listwords = {common.normalize_word(w)
                             for w in str(stim.get(b, "")).split("|") if w}
                avail = set(nev.get(b, set()))
                rescued = set()
                for _, r in g.iterrows():
                    cw = common.normalize_word(r.word)
                    if cw in listwords or r.chosen_prob >= 0.5:
                        continue
                    if r.get("entropy", 0) is not None and float(r.get("entropy", 0)) > HMAX:
                        continue
                    for a in ("alt2", "alt3"):
                        w, p = parse(r.get(a))
                        if w and p >= PMIN and w in listwords and w in avail \
                                and w not in rescued:
                            rescued.add(w)
                            break
                a0 = int(auto.loc[b, "n_correct"])
                a1 = a0 + len(rescued)
                base_a.append(a0)
                resc_a.append(a1)
                hum.append(hmap[b])
                if rescued:
                    n_changed += 1
                    n_rescued_words += len(rescued)
            s0 = agreement_stats(pd.Series(hum), pd.Series(base_a))
            s1 = agreement_stats(pd.Series(hum), pd.Series(resc_a))
            rows.append({"pmin": PMIN, "hmax": HMAX, "n": len(covered),
                         "changed": n_changed, "words_rescued": n_rescued_words,
                         "icc_before": s0["icc21"], "icc_after": s1["icc21"],
                         "mae_before": s0["mae"], "mae_after": s1["mae"],
                         "bias_before": s0["bias_auto_minus_ref"],
                         "bias_after": s1["bias_auto_minus_ref"]})
            print(f"PMIN={PMIN:.2f} HMAX={HMAX:>4}: changed {n_changed:3d} recs "
                  f"({n_rescued_words} words) | ICC {s0['icc21']:.3f} -> {s1['icc21']:.3f}"
                  f" | MAE {s0['mae']:.2f} -> {s1['mae']:.2f}"
                  f" | bias {s0['bias_auto_minus_ref']:+.2f} -> {s1['bias_auto_minus_ref']:+.2f}")
    pd.DataFrame(rows).to_csv(config.OUT_DIR / "rescue_validation.csv", index=False)
    print(f"\nwrote {config.OUT_DIR / 'rescue_validation.csv'}")


if __name__ == "__main__":
    main()
