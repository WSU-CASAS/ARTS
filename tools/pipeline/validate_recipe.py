#!/usr/bin/env python3
"""Recompute every headline number of the frozen recipe (2026-09-freeze).

Reads the current outputs (score tables, token table, calibration bins,
recipe manifest), the transcript cache and the RA hand scores, re-scores the
hand-scored recordings under each configuration and writes

    docs/results/validation_report.md    aggregate tables only
    docs/results/validation_report.json  the same numbers, machine readable

Nothing per recording or per participant is written: the report carries
counts, ICCs and rule-touch totals only.

    python3 tools/pipeline/validate_recipe.py            # rebuild the report
    python3 tools/pipeline/validate_recipe.py --check    # regression guard

--check recomputes the numbers and exits 1 if any of them moved by more than
recipe.CHECK_TOLERANCE from the values recorded in validation_report.json
(without rewriting it). When the hand-score files are absent (CI without
data) the check is skipped with exit 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd

import config
import recipe
from score_wordrecall import prior_words_map, score_recall
from transcribe import load_transcript
from validate import agreement_stats

DOCS_DIR = os.path.join(_REPO, "docs", "results")
REPORT_MD = os.path.join(DOCS_DIR, "validation_report.md")
REPORT_JSON = os.path.join(DOCS_DIR, "validation_report.json")

# Held-out participants (Study 2), from keys/locked_participants.txt or
# SWC_LOCKED_PARTICIPANTS (see config.py). Never used for tuning; excluded from the
# split-half reliability so it describes the development set only.
LOCKED_PARTICIPANTS = config.LOCKED_PARTICIPANTS
SPLIT_HALF_MIN_RECORDINGS = 20

# Token notes -> rule label for the corpus-wide touch table.
RULE_NOTES = [
    ("R2 confusion map", "confusion map:"),
    ("R3 candidate rescue", "rescue:"),
    ("R4 near-miss forcing", "near-miss"),
    ("R5 compound split (one edit)", "compound:"),
    ("R5 compound split (exact)", "compound split"),
    ("R6 recognizer artifact", "recognizer artifact"),
    ("R7 retracted word", "retracted"),
    ("R7 no (never an intrusion)", "no:"),
    ("R8 self-correction", "self-correction"),
    ("R9 outside recall span", "outside recall span"),
    ("R10 repeated intrusion", "repeated intrusion"),
]

CALIBRATION_LABELS = {
    "below 0.20": "< 0.20", "0.20–0.40": "0.20-0.40", "0.40–0.60": "0.40-0.60",
    "0.60–0.80": "0.60-0.80", "0.80–0.95": "0.80-0.95", "0.95 and above": ">= 0.95",
}


# --- hand scores -------------------------------------------------------------
def load_hand_recall() -> pd.DataFrame | None:
    """RA recall scores keyed by basename (Correct, Repetitions, Intrusions,
    Primacy, Middle, Recency, Serial Cluster as numbers)."""
    p = config.HAND_MEM_CSV
    if not p.exists():
        return None
    hs = pd.read_csv(p, header=1, dtype=str)
    hs.columns = [c.strip() for c in hs.columns]
    hs = hs[(hs["Prompt Name"] == "Recall") & hs["Recall Audio File Name"].notna()].copy()
    hs["basename"] = hs["Recall Audio File Name"].str.replace(" ", "", regex=False)
    cols = ["Correct", "Repetitions", "Intrusions", "Primacy", "Middle", "Recency",
            "Serial Cluster"]
    for c in cols:
        hs[c] = pd.to_numeric(hs[c], errors="coerce")
    hs = hs.dropna(subset=cols, how="all").drop_duplicates("basename")
    return hs.set_index("basename")[cols]


# --- helpers ---------------------------------------------------------------
def stats(ref, auto) -> dict:
    """agreement_stats trimmed to what the report tables show."""
    s = agreement_stats(ref, auto)
    return {"n": int(s["n"]), "icc": s.get("icc21"), "mae": s.get("mae"),
            "bias": s.get("bias_auto_minus_ref"), "exact": s.get("exact_match_rate"),
            "within1": s.get("within1_rate")}


def toward_away(ref, base, new) -> dict:
    """How many recordings a configuration changed relative to a baseline,
    and whether each change moved toward or away from the RA value."""
    ref, base, new = (np.asarray(x, float) for x in (ref, base, new))
    changed = new != base
    toward = changed & (np.abs(new - ref) < np.abs(base - ref))
    away = changed & (np.abs(new - ref) > np.abs(base - ref))
    return {"changed": int(changed.sum()), "toward": int(toward.sum()),
            "away": int(away.sum()), "same_distance": int((changed & ~toward & ~away).sum())}


def fmt(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# --- recall ------------------------------------------------------------------
def recall_section(hand: pd.DataFrame) -> dict:
    """Recall correct under legacy / automatic / ra_informed, intrusions and
    repetitions under both conventions and both R9 stop lists."""
    idx = pd.read_csv(config.INDEX_CSV)
    idx = idx[(idx["task"] == "word_recall") & idx["audio_found"]]
    priors = prior_words_map(idx)
    stim = dict(zip(idx["basename"], idx["stimulus"]))
    rows = []
    for b in hand.index:
        if b not in stim:
            continue
        tr = load_transcript(b)
        if tr is None:
            continue
        targets = [t for t in str(stim[b]).split("|") if t]
        row = {"basename": b}
        leg = score_recall(tr, targets, prior_words=priors.get(b), basename=b, legacy=True)
        row.update(legacy_correct=leg["n_correct"], legacy_intrusions=leg["n_intrusions"],
                   legacy_repetitions=leg["n_repetitions"])
        for cfg in recipe.RECIPE_CONFIGS:
            for span in recipe.SPAN_STOP_VARIANTS:
                s = score_recall(tr, targets, prior_words=priors.get(b), basename=b,
                                 recipe_config=cfg, span_stop=span)
                row[f"{cfg}_{span}_correct"] = s["n_correct"]
                row[f"{cfg}_{span}_intrusions"] = s["n_intrusions"]
                row[f"{cfg}_{span}_intrusions_all"] = s["n_intrusions_all"]
                row[f"{cfg}_{span}_intrusions_span_only"] = s["n_intrusions_in_span"]
                row[f"{cfg}_{span}_repetitions"] = s["n_repetitions"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("basename").join(hand, how="inner")
    out = {"n_transcribed_hand_scored": len(df)}

    cor = df.dropna(subset=["Correct"])
    d = recipe.SPAN_STOP_DEFAULT
    out["correct"] = {
        "legacy_baseline": stats(cor["Correct"], cor["legacy_correct"]),
        "automatic": {**stats(cor["Correct"], cor[f"automatic_{d}_correct"]),
                      **toward_away(cor["Correct"], cor["legacy_correct"],
                                    cor[f"automatic_{d}_correct"])},
        "ra_informed": {**stats(cor["Correct"], cor[f"ra_informed_{d}_correct"]),
                        **toward_away(cor["Correct"], cor["legacy_correct"],
                                      cor[f"ra_informed_{d}_correct"])},
    }
    # the production table must agree with the recomputation for its configuration
    prod_path = config.OUT_DIR / "scores_word_recall.csv"
    if prod_path.exists():
        prod = pd.read_csv(prod_path).set_index("basename")
        pcfg = str(prod["recipe_config"].iloc[0]) if "recipe_config" in prod else "unknown"
        j = cor.join(prod[["n_correct", "n_intrusions", "n_repetitions"]], how="inner")
        key = f"{pcfg}_{d}" if pcfg in recipe.RECIPE_CONFIGS else None
        out["correct"]["production_table"] = {
            "recipe_config": pcfg,
            "recipe_version": str(prod["recipe_version"].iloc[0]) if "recipe_version" in prod else "",
            **stats(j["Correct"], j["n_correct"]),
            "rows_differing_from_recomputation": int((j["n_correct"] != j[f"{key}_correct"]).sum())
            if key else None,
        }

    intr = df.dropna(subset=["Intrusions"])
    rep = df.dropna(subset=["Repetitions"])
    out["intrusions"] = {"legacy_all_tokens": stats(intr["Intrusions"], intr["legacy_intrusions"])}
    out["repetitions"] = {"legacy": stats(rep["Repetitions"], rep["legacy_repetitions"])}
    for cfg in recipe.RECIPE_CONFIGS:
        out["intrusions"][f"{cfg}_all_tokens"] = stats(
            intr["Intrusions"], intr[f"{cfg}_fillers_intrusions_all"])
        for span in recipe.SPAN_STOP_VARIANTS:
            out["intrusions"][f"{cfg}_r9_{span}_r10"] = stats(
                intr["Intrusions"], intr[f"{cfg}_{span}_intrusions"])
            out["intrusions"][f"{cfg}_r9_{span}_without_r10"] = stats(
                intr["Intrusions"], intr[f"{cfg}_{span}_intrusions_span_only"])
        out["repetitions"][cfg] = stats(rep["Repetitions"], rep[f"{cfg}_fillers_repetitions"])
    gain = (out["intrusions"]["ra_informed_r9_chatter_r10"]["icc"]
            - out["intrusions"]["ra_informed_r9_fillers_r10"]["icc"])
    out["span_stop"] = {
        "frozen_default": recipe.SPAN_STOP_DEFAULT,
        "margin": recipe.CHATTER_ICC_MARGIN,
        "chatter_minus_fillers_icc_ra_informed": round(float(gain), 3),
        "rule_selects": "chatter" if gain > recipe.CHATTER_ICC_MARGIN else "fillers",
    }
    out["span_stop"]["consistent"] = out["span_stop"]["rule_selects"] == recipe.SPAN_STOP_DEFAULT
    return out


def serial_section(hand: pd.DataFrame) -> dict:
    """Serial-position family from outputs/scores_serial.csv (unchanged measures)."""
    p = config.OUT_DIR / "scores_serial.csv"
    if not p.exists():
        return {"missing": str(p.name)}
    ser = pd.read_csv(p).set_index("basename")
    j = hand.join(ser, how="inner")
    out = {}
    for label, hcol, acol in [("primacy", "Primacy", "primacy_3_6_3"),
                              ("middle", "Middle", "middle_3_6_3"),
                              ("recency", "Recency", "recency_3_6_3"),
                              ("serial_cluster", "Serial Cluster", "serial_cluster")]:
        s = j.dropna(subset=[hcol, acol])
        out[label] = stats(s[hcol], s[acol])
    return out


# --- corpus-wide rule touches ----------------------------------------------
def rule_touch_section() -> dict:
    tok_path = config.OUT_DIR / "recall_tokens_long.csv"
    out = {}
    if tok_path.exists():
        tok = pd.read_csv(tok_path, usecols=["basename", "classification", "note"])
        notes = tok["note"].fillna("").astype(str)
        rec = tok["basename"]
        out["recall"] = {"recordings": int(rec.nunique()), "tokens": len(tok), "rules": {}}
        for label, prefix in RULE_NOTES:
            mask = notes.str.contains(prefix, regex=False)
            out["recall"]["rules"][label] = {"tokens": int(mask.sum()),
                                             "recordings": int(rec[mask].nunique())}
        out["recall"]["classification_counts"] = {
            k: int(v) for k, v in Counter(tok["classification"]).items()}
    return out


# --- calibration and split-half ---------------------------------------------
def calibration_section() -> dict:
    p = config.OUT_DIR / "word_confidence_calibration.csv"   # tools/pipeline/confidence_calibration.py
    if not p.exists():
        return {"missing": p.name}
    cal = pd.read_csv(p)
    bins = []
    for _, r in cal.iterrows():
        bins.append({"bin": CALIBRATION_LABELS.get(str(r["bin"]), str(r["bin"])),
                     "words": int(r["words"]), "accuracy_pct": float(r["accuracy"])})
    total = sum(b["words"] for b in bins)
    overall = sum(b["words"] * b["accuracy_pct"] for b in bins) / total if total else float("nan")
    return {"bins": bins, "words": total, "overall_accuracy_pct": round(overall, 1)}


def split_half_section() -> dict:
    """Person-level odd/even split-half reliability of the machine scores on the
    development participants (locked ones excluded): mean correct, mean
    intrusions per recording, mean repetitions per recording; Pearson r
    between halves and its Spearman-Brown correction 2r/(1+r)."""
    p = config.OUT_DIR / "scores_word_recall.csv"
    if not p.exists():
        return {"missing": p.name}
    sc = pd.read_csv(p)
    sc = sc[~sc["participant_num"].isin(LOCKED_PARTICIPANTS)].copy()
    sc["dt"] = pd.to_datetime(sc["datetime"], errors="coerce")
    sc = sc.dropna(subset=["dt", "participant_num"]).sort_values(["participant_num", "dt"])
    measures = {"correct": "n_correct", "intrusion_rate": "n_intrusions",
                "repetition_rate": "n_repetitions"}
    halves = {m: ([], []) for m in measures}
    n_participants = 0
    sizes = []
    for _, g in sc.groupby("participant_num"):
        if len(g) < SPLIT_HALF_MIN_RECORDINGS:
            continue
        n_participants += 1
        sizes.append(len(g))
        odd, even = g.iloc[::2], g.iloc[1::2]
        for m, col in measures.items():
            halves[m][0].append(odd[col].mean())
            halves[m][1].append(even[col].mean())
    out = {"participants": n_participants,
           "min_recordings": SPLIT_HALF_MIN_RECORDINGS,
           "median_recordings": float(np.median(sizes)) if sizes else None,
           "locked_participants_excluded": len(LOCKED_PARTICIPANTS)}
    for m in measures:
        a, b = (np.asarray(x, float) for x in halves[m])
        r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() and b.std() else float("nan")
        out[m] = {"r": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3)}
    return out


# --- flatten for --check ----------------------------------------------------
def flatten(d, prefix="") -> dict:
    """Every numeric leaf of the report as key -> float (the --check surface)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(flatten(item, f"{key}[{i}]."))
        elif isinstance(v, bool) or v is None:
            continue
        elif isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(v):
            out[key] = float(v)
    return out


# --- report ------------------------------------------------------------------
def build_report() -> dict:
    hand = load_hand_recall()
    report = {
        "recipe_version": recipe.RECIPE_VERSION,
        "recipe_config_default": config.RECIPE_CONFIG,
        "thresholds": recipe.THRESHOLDS,
        "manifest": recipe.read_manifest(),
        "hand_scores_available": hand is not None,
    }
    if hand is not None:
        report["recall"] = recall_section(hand)
        report["serial"] = serial_section(hand)
    report["rule_touches"] = rule_touch_section()
    report["calibration"] = calibration_section()
    report["split_half"] = split_half_section()
    return report


def _stat_row(label, s, extra=()):
    return [label, s.get("n"), fmt(s.get("icc")), fmt(s.get("mae"), 2), fmt(s.get("bias"), 2),
            fmt(s.get("exact"), 2), fmt(s.get("within1"), 2), *extra]


def render_md(rep: dict) -> str:
    L = []
    L += [f"# Validation report: frozen recipe {rep['recipe_version']}", "",
          "Generated by `tools/pipeline/validate_recipe.py` from the current outputs, the "
          "transcript cache and the RA hand scores. Development set only (Study 1); the "
          "held-out participants are evaluated separately by `study2_reliability.py`. "
          "All agreement statistics are ICC(2,1) with MAE and bias (auto minus RA) from "
          "`validate.agreement_stats`. Aggregates only.", ""]
    m = rep.get("manifest") or {}
    L += ["## Recipe manifest", "",
          f"- recipe version: `{m.get('recipe_version', 'not written')}`",
          f"- configuration: `{m.get('recipe_config', config.RECIPE_CONFIG)}` "
          f"(default {config.RECIPE_CONFIG}; `automatic` disables the confusion map R2 only)",
          f"- git commit: `{m.get('git_commit')}`",
          f"- generated: {m.get('generated_at')}",
          f"- steps recorded: {', '.join(sorted((m.get('steps') or {}).keys())) or 'none'}", "",
          "Thresholds:", ""]
    L += [table(["threshold", "value"], [[k, v] for k, v in rep["thresholds"].items()]), ""]
    if not rep["hand_scores_available"]:
        L += ["Hand-score files were not available; agreement sections are omitted.", ""]

    H = ["variant", "n", "ICC(2,1)", "MAE", "bias", "exact", "within 1"]
    if "recall" in rep:
        r = rep["recall"]
        L += ["## Recall: correct words (RA `Correct`)", ""]
        rows = [_stat_row("legacy baseline (pre-freeze scorer)", r["correct"]["legacy_baseline"],
                          ["", "", ""])]
        for cfg in ("automatic", "ra_informed"):
            s = r["correct"][cfg]
            rows.append(_stat_row(f"frozen, {cfg}", s, [s["changed"], s["toward"], s["away"]]))
        L += [table(H + ["changed vs baseline", "toward RA", "away from RA"], rows), ""]
        pt = r["correct"].get("production_table")
        if pt:
            L += [f"Production table `outputs/scores_word_recall.csv` (config `{pt['recipe_config']}`, "
                  f"version `{pt['recipe_version']}`): ICC {fmt(pt['icc'])}, MAE {fmt(pt['mae'], 2)}, "
                  f"n {pt['n']}; rows differing from the recomputation: "
                  f"{pt['rows_differing_from_recomputation']}.", ""]
        L += ["## Recall: intrusions (RA `Intrusions`)", "",
              "Old convention counts every off-list token. The frozen convention counts "
              "intrusion tokens inside the recall span only (R9) and adds repeated intrusions "
              "(R10, decision D2). Both R9 stop lists are reported; the frozen default is "
              f"`{r['span_stop']['frozen_default']}` (rule: fillers only unless the chatter list "
              f"improves the ICC by more than {r['span_stop']['margin']}; observed gain "
              f"{r['span_stop']['chatter_minus_fillers_icc_ra_informed']:+.3f}, rule selects "
              f"`{r['span_stop']['rule_selects']}`, consistent: {r['span_stop']['consistent']}).", ""]
        rows = [_stat_row("legacy, all tokens", r["intrusions"]["legacy_all_tokens"])]
        for cfg in ("automatic", "ra_informed"):
            rows.append(_stat_row(f"{cfg}, all tokens (old convention)", r["intrusions"][f"{cfg}_all_tokens"]))
            for span in recipe.SPAN_STOP_VARIANTS:
                mark = " (frozen)" if span == recipe.SPAN_STOP_DEFAULT and cfg == "ra_informed" else ""
                rows.append(_stat_row(f"{cfg}, R9 {span} + R10{mark}",
                                      r["intrusions"][f"{cfg}_r9_{span}_r10"]))
                rows.append(_stat_row(f"{cfg}, R9 {span} without R10",
                                      r["intrusions"][f"{cfg}_r9_{span}_without_r10"]))
        L += [table(H, rows), ""]
        L += ["## Recall: repetitions (RA `Repetitions`)", "",
              "n_repetitions counts repeats of credited words and repeated intrusions in both "
              "conventions (D2 changes the intrusion count, not the repetition count); the "
              "frozen scorer additionally exempts words named in a self-correction statement "
              "(R8) and retracted words (R7).", ""]
        rows = [_stat_row("legacy", r["repetitions"]["legacy"])]
        rows += [_stat_row(f"frozen, {cfg}", r["repetitions"][cfg]) for cfg in ("automatic", "ra_informed")]
        L += [table(H, rows), ""]
    if "serial" in rep and "missing" not in rep["serial"]:
        L += ["## Serial position family (R13, unchanged measures)", "",
              "From `outputs/scores_serial.csv` (3-6-3 split) against the RA columns.", ""]
        L += [table(H, [_stat_row(k, v) for k, v in rep["serial"].items()]), ""]
    rt = rep.get("rule_touches", {})
    if "recall" in rt:
        rc = rt["recall"]
        L += ["## Rule touches, corpus-wide", "",
              f"Recall token table (`outputs/recall_tokens_long.csv`): {rc['recordings']} recordings, "
              f"{rc['tokens']} rows (spoken tokens plus never-recalled targets).", ""]
        L += [table(["rule", "tokens", "recordings"],
                    [[k, v["tokens"], v["recordings"]] for k, v in rc["rules"].items()]), ""]
        L += [table(["classification", "rows"],
                    [[k, v] for k, v in sorted(rc["classification_counts"].items())]), ""]
    cal = rep.get("calibration", {})
    if "bins" in cal:
        L += ["## Word confidence calibration", "",
              f"Whisper word probability against hand transcripts ({cal['words']} words; overall "
              f"accuracy {cal['overall_accuracy_pct']:.1f} percent).", ""]
        L += [table(["probability bin", "words", "accuracy (percent)"],
                    [[b["bin"], b["words"], f"{b['accuracy_pct']:.1f}"] for b in cal["bins"]]), ""]
    sh = rep.get("split_half", {})
    if "participants" in sh:
        L += ["## Person-level split-half reliability (development participants)", "",
              f"Odd/even interleave of each participant's recall recordings in time order; "
              f"{sh['participants']} participants with at least {sh['min_recordings']} recordings "
              f"(median {fmt(sh['median_recordings'], 0)}); {sh['locked_participants_excluded']} "
              "locked participants excluded. Person means per half, Pearson r between halves, "
              "Spearman-Brown corrected.", ""]
        L += [table(["measure", "r", "Spearman-Brown"],
                    [[k, fmt(sh[k]["r"]), fmt(sh[k]["spearman_brown"])]
                     for k in ("correct", "intrusion_rate", "repetition_rate")]), ""]
    return "\n".join(L)


def check(rep: dict) -> int:
    """Compare the recomputed numbers with the recorded report; 1 on drift."""
    if not os.path.exists(REPORT_JSON):
        print(f"[validate_recipe] no recorded report at {REPORT_JSON}; nothing to check")
        return 0
    with open(REPORT_JSON) as fh:
        recorded = json.load(fh)
    old, new = flatten(recorded), flatten(rep)
    skip = ("manifest.",)  # timestamps, hashes and step counts are not agreement numbers
    drift = []
    for k, v in old.items():
        if k.startswith(skip):
            continue
        if k not in new:
            drift.append((k, v, None))
        elif abs(new[k] - v) > recipe.CHECK_TOLERANCE + 1e-12:
            drift.append((k, v, new[k]))
    if drift:
        print(f"[validate_recipe] {len(drift)} number(s) moved more than {recipe.CHECK_TOLERANCE}:")
        for k, a, b in drift:
            print(f"  {k}: recorded {a} -> now {b}")
        return 1
    print(f"[validate_recipe] check passed: {len(old)} numbers within {recipe.CHECK_TOLERANCE}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="compare with the recorded report and exit 1 on drift; write nothing")
    args = ap.parse_args(argv)
    if not config.HAND_MEM_CSV.exists():
        if args.check:
            print("[validate_recipe] hand-score files not available; check skipped")
            return 0
        print("[validate_recipe] WARNING: hand-score files not available; agreement sections omitted")
    rep = build_report()
    if args.check:
        return check(rep)
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(REPORT_JSON, "w") as fh:
        json.dump(rep, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    md = render_md(rep)
    assert "\u2014" not in md, "long dash in report"
    with open(REPORT_MD, "w") as fh:
        fh.write(md + "\n")
    print(f"[validate_recipe] wrote {REPORT_MD} and {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
