#!/usr/bin/env python3
"""Did word_timestamps change the decode, or is Whisper just nondeterministic?

Three caches over the SAME 200 recordings:

  A  cache/transcripts             the original run   (no word_timestamps)
  B  cache/transcripts_words       the timing run     (word_timestamps=True)
  C  cache/transcripts_control     a plain re-run     (no word_timestamps)

A vs B is what we already measured (19% of the corpus differed). A vs C is the
control: the ONLY thing that changed is that it ran a second time.

  drift(A,C) close to drift(A,B)  ->  plain nondeterminism; the flag is innocent
  drift(A,C) close to zero        ->  word_timestamps altered the decoding path

The sharper test is the stratified one. Of the 100 sampled files that drifted
under B, how many also drift under C? If most do, those recordings are simply
unstable and no setting will fix them.

    python3 tools/analysis/drift_control_analysis.py            # uses default paths
    python3 tools/analysis/drift_control_analysis.py --a ... --b ... --c ... --sample ...

Read-only: writes nothing unless --report-json is given.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s']")


def norm(s: str) -> str:
    """Case/punctuation/whitespace-insensitive form, for a fair comparison."""
    return _WS.sub(" ", _PUNCT.sub(" ", (s or "").lower())).strip()


def load_text(cache: str, basename: str) -> str | None:
    p = os.path.join(cache, basename + ".json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get("text", "")
    except Exception:
        return None


def wer(ref: str, hyp: str) -> float:
    """Levenshtein over word tokens, normalised by reference length."""
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def compare(a_cache, x_cache, names):
    """Return (n_compared, n_differ, mean_wer, differing_names)."""
    n = differ = 0
    tot = 0.0
    diffs = []
    for b in names:
        ta, tx = load_text(a_cache, b), load_text(x_cache, b)
        if ta is None or tx is None:
            continue
        n += 1
        na, nx = norm(ta), norm(tx)
        tot += wer(na, nx)
        if na != nx:
            differ += 1
            diffs.append(b)
    return n, differ, (tot / n if n else 0.0), diffs


def pct(k, n):
    return f"{100.0 * k / n:.1f}%" if n else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="cache/transcripts", help="original run")
    ap.add_argument("--b", default="cache/transcripts_words", help="word_timestamps run")
    ap.add_argument("--c", default="cache/transcripts_control", help="plain re-run")
    ap.add_argument("--sample", default="outputs/driftexp/control_sample.txt")
    ap.add_argument("--strata", default="outputs/driftexp/control_strata.json")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    if not os.path.isdir(args.c):
        sys.exit(f"control cache not found: {args.c}\n"
                 f"Run the control pass first (see the docstring).")

    names = [ln.strip() for ln in open(args.sample) if ln.strip()]
    strata = json.load(open(args.strata)) if os.path.exists(args.strata) else {}
    drifted_b = set(strata.get("drifted_in_sample", []))
    same_b = set(strata.get("same_in_sample", []))

    print(f"sample: {len(names)} recordings "
          f"({len(drifted_b)} drifted under word_timestamps, {len(same_b)} did not)\n")

    nab, dab, wab, _ = compare(args.a, args.b, names)
    nac, dac, wac, diffs_ac = compare(args.a, args.c, names)

    print("--- overall on the sample ---------------------------------")
    print(f"  A vs B  (word_timestamps run) : {dab}/{nab} differ  ({pct(dab, nab)})   mean WER {wab:.4f}")
    print(f"  A vs C  (plain re-run)        : {dac}/{nac} differ  ({pct(dac, nac)})   mean WER {wac:.4f}")

    set_ac = set(diffs_ac)
    # Denominators must be the files actually re-run so far, NOT the full
    # stratum, or a partial control pass silently understates every rate.
    done_d = {b for b in drifted_b if load_text(args.c, b) is not None}
    done_s = {b for b in same_b if load_text(args.c, b) is not None}
    print("\n--- the stratified test -----------------------------------")
    if done_d:
        also = len(done_d & set_ac)
        print(f"  of the {len(done_d)} re-run that drifted under B, "
              f"{also} ({pct(also, len(done_d))}) also drift on a plain re-run")
    if done_s:
        newly = len(done_s & set_ac)
        print(f"  of the {len(done_s)} re-run that were stable under B, "
              f"{newly} ({pct(newly, len(done_s))}) drift on a plain re-run")
    if done_d:
        share = 1.0 - (len(done_d & set_ac) / len(done_d))
        print(f"\n  -> {share*100:.0f}% of the drift under B is attributable to the FLAG;"
              f" {100-share*100:.0f}% is unstable audio")

    print("\n--- verdict -----------------------------------------------")
    MIN_N, MIN_STRATUM = 40, 15
    n_drift_done = len([b for b in drifted_b if load_text(args.c, b) is not None])
    if nac < MIN_N or n_drift_done < MIN_STRATUM:
        print("  INCONCLUSIVE, not enough of the control run has completed.")
        print(f"  Compared {nac} recordings ({n_drift_done} from the drifted stratum).")
        print(f"  Need at least {MIN_N} overall and {MIN_STRATUM} drifted before this")
        print("  distinguishes anything. Finish the control pass and re-run.")
    elif dac == 0:
        print("  WORD_TIMESTAMPS CHANGED THE DECODE.")
        print("  A plain re-run reproduced the original text exactly, so the flag is")
        print("  responsible for the drift, not run-to-run randomness.")
    elif abs((dac / nac) - (dab / nab if nab else 0)) <= 0.10:
        print("  PLAIN NONDETERMINISM.")
        print("  A re-run with identical settings drifts about as much as the timing run,")
        print("  so the flag is innocent. Whisper simply is not reproducible on this audio.")
        print("  Reportable as an instrument-reliability property, not a bug.")
    else:
        print("  MIXED. Both effects appear to contribute; report both rates side by side.")

    if args.report_json:
        json.dump({"n": len(names),
                   "a_vs_b": {"n": nab, "differ": dab, "mean_wer": wab},
                   "a_vs_c": {"n": nac, "differ": dac, "mean_wer": wac},
                   "drifted_under_b_also_under_c": sorted(drifted_b & set_ac),
                   "stable_under_b_but_drift_under_c": sorted(same_b & set_ac)},
                  open(args.report_json, "w"), indent=1)
        print(f"\nwrote {args.report_json}")


if __name__ == "__main__":
    main()
