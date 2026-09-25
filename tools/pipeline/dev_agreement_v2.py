#!/usr/bin/env python3
"""Development-set agreement of a second recipe version against the frozen one.

Reads two recall score tables (the frozen outputs/scores_word_recall.csv and a
run under another version, e.g. outputs/v2/scores_word_recall.csv), joins both
with the RA hand scores on the development recordings (locked Study 2
participants excluded, as in validate_recipe.py) and writes aggregate
agreement (ICC(2,1), MAE, bias, exact, within-1) for correct, repetitions and
intrusions under each version, the toward/away counts of every change, and
corpus-wide touch counts of the version's rules, split into development and
locked recordings. Aggregates only; nothing per recording is written.

    python3 tools/pipeline/dev_agreement_v2.py --scores outputs/v2/scores_word_recall.csv \\
        --out outputs/v2/dev_agreement.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

import pandas as pd

import config
from validate_recipe import LOCKED_PARTICIPANTS, load_hand_recall, stats, toward_away

MEASURES = [("correct", "Correct", "n_correct"),
            ("repetitions", "Repetitions", "n_repetitions"),
            ("intrusions", "Intrusions", "n_intrusions")]
TOUCH_COLS = ["n_loop_guard_dropped", "n_intrusions_narration"]


def _version(df: pd.DataFrame) -> str:
    return str(df["recipe_version"].iloc[0]) if "recipe_version" in df else "unknown"


def build(frozen_path: str, new_path: str) -> dict:
    hand = load_hand_recall()
    if hand is None:
        raise SystemExit(f"hand scores not found at {config.HAND_MEM_CSV}")
    frozen = pd.read_csv(frozen_path).set_index("basename")
    new = pd.read_csv(new_path).set_index("basename")
    common_idx = frozen.index.intersection(new.index)
    frozen, new = frozen.loc[common_idx], new.loc[common_idx]
    locked = new["participant_num"].isin(LOCKED_PARTICIPANTS)
    out = {"frozen_version": _version(frozen), "new_version": _version(new),
           "n_recordings": int(len(common_idx)),
           "n_development": int((~locked).sum()), "n_locked": int(locked.sum())}

    dev = new[~locked].join(hand, how="inner")
    out["development"] = {}
    for key, hcol, mcol in MEASURES:
        d = dev.dropna(subset=[hcol])
        ref, base, cur = d[hcol], frozen.loc[d.index, mcol], d[mcol]
        out["development"][key] = {
            "frozen": stats(ref, base),
            "new": {**stats(ref, cur), **toward_away(ref, base, cur)},
        }
    # corpus-wide changes of the score columns and rule touch counts
    changes = {}
    for key, _h, mcol in MEASURES:
        diff = (new[mcol] != frozen[mcol])
        changes[key] = {"development": int(diff[~locked].sum()), "locked": int(diff[locked].sum())}
    out["score_changes"] = changes
    touches = {}
    for col in TOUCH_COLS:
        if col not in new:
            continue
        v = new[col].fillna(0)
        touches[col] = {
            "development": {"recordings": int((v[~locked] > 0).sum()), "tokens": int(v[~locked].sum())},
            "locked": {"recordings": int((v[locked] > 0).sum()), "tokens": int(v[locked].sum())},
        }
    out["rule_touches"] = touches
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frozen", default=os.path.join(_REPO, "outputs", "scores_word_recall.csv"))
    ap.add_argument("--scores", required=True, help="score table of the version to compare")
    ap.add_argument("--out", default=None, help="JSON destination (default: beside --scores)")
    args = ap.parse_args(argv)
    rep = build(args.frozen, args.scores)
    dst = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores)), "dev_agreement.json")
    with open(dst, "w") as fh:
        json.dump(rep, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for key in ("correct", "repetitions", "intrusions"):
        f, n = rep["development"][key]["frozen"], rep["development"][key]["new"]
        print(f"[dev_agreement] {key:12s} n={f['n']}  frozen ICC {f['icc']:.3f} MAE {f['mae']:.2f} "
              f"bias {f['bias']:+.2f}  ->  {rep['new_version']} ICC {n['icc']:.3f} MAE {n['mae']:.2f} "
              f"bias {n['bias']:+.2f}  changed {n['changed']} toward {n['toward']} away {n['away']}")
    print(f"[dev_agreement] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
