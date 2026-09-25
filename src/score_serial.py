"""Serial-order features for word-list recall.

Scored against the FIXED presentation order the participant saw during the
learning prompts (keys/word_lists_ordered.json, from Bryan Minor 2026-07-13).

IMPORTANT: that order only applies to participants who started May 2025 or
later (protocol v2). Earlier pilot participants used a different system, so
serial-order features are not emitted for them.

Features (per Dr. Cook's spec, 2026-07-13):
  serial position   n_correct in primacy / middle / recency, BOTH splits
                      split_4_4_4 : 1-4 / 5-8 / 9-12
                      split_3_6_3 : 1-3 / 4-9 / 10-12   (what the RAs used)
  serial recall     four versions, each comparing the recalled order to the
                    presentation order:
                      pairwise_order_accuracy : fraction of recalled word pairs
                                                in the same relative order as shown
                      lcs                     : longest common subsequence length
                      serial_position_accuracy: words recalled at their exact
                                                presentation position
                      edit_distance           : Levenshtein of recalled sequence
                                                vs the studied order (recalled
                                                words only)
  serial_cluster    consecutive recalls adjacent in presentation order
                    (forward, diff == +1) -- matches the RA "Serial Cluster"

    python src/score_serial.py
    python src/score_serial.py --out-dir outputs/v2   # reads and writes there
"""
from __future__ import annotations
import _bootstrap  # noqa: F401  (path shim, must be first)
import argparse
import json
import re
from pathlib import Path
from itertools import combinations

import pandas as pd

import config
import common
import recipe
from score_wordrecall import classify_tokens, resolve_out_dir

V2_START = pd.Timestamp("2025-05-01")  # canonical order applies from here on


def load_order() -> dict[int, list[str]]:
    raw = json.loads((config.KEYS_DIR / "word_lists_ordered.json").read_text())
    return {n: raw[f"list_{n}"]["memory"] for n in range(1, 7) if f"list_{n}" in raw}


def list_num(list_id) -> int | None:
    m = re.match(r"list_(\d+)", str(list_id).strip())
    return int(m.group(1)) if m else None


def recalled_positions(transcript: str, order: list[str], prior_words=None,
                       **kw) -> list[int]:
    """1-based presentation positions of correctly recalled words, in spoken
    order. Keyword arguments (basename, legacy, ...) go to classify_tokens so
    the credited words are the same ones the recall score counts."""
    pos = {w: i + 1 for i, w in enumerate(order)}
    tokens, _ = classify_tokens(transcript, list(order), lenient=True,
                                prior_words=prior_words, **kw)
    return [pos[t["matched_list_word"]] for t in tokens
            if t["classification"] == "correct" and t["matched_list_word"] in pos]


# ---- serial position -------------------------------------------------------

def serial_position(seq: list[int]) -> dict:
    def band(lo, hi):
        return sum(1 for x in seq if lo <= x <= hi)
    return {
        "primacy_4_4_4": band(1, 4), "middle_4_4_4": band(5, 8), "recency_4_4_4": band(9, 12),
        "primacy_3_6_3": band(1, 3), "middle_3_6_3": band(4, 9), "recency_3_6_3": band(10, 12),
    }


# ---- serial recall, four versions -----------------------------------------

def pairwise_order_accuracy(seq: list[int]) -> float | None:
    """Of all pairs of recalled words, the fraction spoken in the same relative
    order as they were presented. 1.0 = perfect order, 0.5 = chance, None if <2."""
    if len(seq) < 2:
        return None
    pairs = list(combinations(range(len(seq)), 2))
    ok = sum(1 for i, j in pairs if seq[i] < seq[j])
    return ok / len(pairs)


def lcs_length(seq: list[int]) -> int:
    """Longest run (not necessarily contiguous) recalled in presentation order."""
    if not seq:
        return 0
    best = []  # patience sorting / LIS in O(n log n)
    import bisect
    for x in seq:
        i = bisect.bisect_left(best, x)
        if i == len(best):
            best.append(x)
        else:
            best[i] = x
    return len(best)


def serial_position_accuracy(seq: list[int]) -> int:
    """Words recalled at their exact presentation position (1st said = word 1, ...)."""
    return sum(1 for i, x in enumerate(seq) if x == i + 1)


def edit_distance_to_order(seq: list[int]) -> int:
    """Levenshtein between the recalled sequence and those same words in
    presentation order. 0 = recalled in perfect relative order."""
    return common._levenshtein(seq, sorted(seq))


def serial_cluster(seq: list[int]) -> int:
    """Consecutive recalls that were adjacent-and-forward in presentation order."""
    return sum(1 for a, b in zip(seq, seq[1:]) if b - a == 1)


def all_serial_features(seq: list[int]) -> dict:
    f = {"n_correct_ordered": len(seq), "recalled_positions": "|".join(map(str, seq))}
    f.update(serial_position(seq))
    f.update({
        "pairwise_order_accuracy": pairwise_order_accuracy(seq),
        "lcs": lcs_length(seq),
        "serial_position_accuracy": serial_position_accuracy(seq),
        "edit_distance": edit_distance_to_order(seq),
        "serial_cluster": serial_cluster(seq),
    })
    return f


# ---- driver ----------------------------------------------------------------

def v2_devices(idx: pd.DataFrame) -> set:
    """Devices whose first recall is on/after the v2 cutover."""
    d = idx.dropna(subset=["datetime"])
    first = d.groupby("device")["datetime"].min()
    return set(first[first >= V2_START].index)


def main(argv=None):
    ap = argparse.ArgumentParser(description="serial-order features under the current recipe version")
    ap.add_argument("--out-dir", default=None,
                    help="directory holding scores_word_recall.csv and receiving scores_serial.csv "
                         "(default SWC_OUT_DIR or outputs/)")
    ap.add_argument("--scores", default=None,
                    help="recall score table to read (default <out-dir>/scores_word_recall.csv)")
    args = ap.parse_args(argv)
    out_dir = resolve_out_dir(args.out_dir)
    scores = pd.read_csv(Path(args.scores) if args.scores else out_dir / "scores_word_recall.csv")
    idx = pd.read_csv(config.INDEX_CSV, parse_dates=["datetime"])
    order = load_order()

    keep = v2_devices(idx)
    dev = dict(zip(idx["basename"], idx["device"]))
    scores["device"] = scores["basename"].map(dev)

    rows = []
    skipped_v1 = skipped_nolist = 0
    for _, r in scores.iterrows():
        if r["device"] not in keep:
            skipped_v1 += 1
            continue
        n = list_num(r.get("list_id"))
        if n not in order:
            skipped_nolist += 1
            continue
        seq = recalled_positions(r.get("transcript"), order[n], basename=r["basename"])
        rows.append({"basename": r["basename"], "device": r["device"],
                     "datetime": r.get("datetime"), "list_id": r.get("list_id"),
                     "recipe_version": recipe.RECIPE_VERSION,
                     "recipe_config": config.RECIPE_CONFIG,
                     **all_serial_features(seq)})
    out = pd.DataFrame(rows)
    dst = out_dir / "scores_serial.csv"
    out.to_csv(dst, index=False)
    print(f"[serial] scored {len(out)} v2 trials -> {dst}")
    recipe.write_manifest("serial", {"n_scored": int(len(out)), "output": dst.name},
                          manifest_path=out_dir / "recipe_manifest.json")
    readme = recipe.write_exploratory_readme(out_dir)
    if readme:
        print(f"[serial] EXPLORATORY run under recipe {recipe.RECIPE_VERSION}; marker -> {readme}")
    print(f"[serial] skipped: {skipped_v1} v1/pre-May-2025 trials, {skipped_nolist} without a usable list id")
    if len(out):
        print(out[["primacy_3_6_3", "middle_3_6_3", "recency_3_6_3",
                   "pairwise_order_accuracy", "lcs", "serial_position_accuracy",
                   "edit_distance", "serial_cluster"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
