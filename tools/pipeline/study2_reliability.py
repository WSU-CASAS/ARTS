"""Study 2: reliability of the frozen recipe against two blind human passes.

Given the two completed rating workbooks (pass A = Rater 1, pass B = Rater 2; tabs
"Transcriptions" and "Score - word recall", columns in REQUIRED_COLS below) this
script computes, for every recall measure, three rows of agreement:

    machine vs pass A, machine vs pass B, pass A vs pass B (the human ceiling),

plus a machine vs mean-of-humans row, each with ICC(2,1), MAE, bias and n from
validate.agreement_stats. Transcript-level agreement is the word error rate
(common.wer) between the two human transcriptions (blind human-human) and
between the machine transcript and each pass.

Rows are matched on the audio file name with spaces removed. Blank and
non-numeric cells are tolerated, dropped from the pair and counted in the
report. Rows the humans have not scored yet (every score cell blank) are
skipped, so a partially completed workbook already gives a first estimate.

Outputs (default outputs/study2/, gitignored):
    study2_score_agreement.csv       per-measure agreement table
    study2_transcript_agreement.csv  WER table per task
    study2_paired_scores.csv         per-clip paired values (participant data,
                                     stays under outputs/)
    study2_paired_transcripts.csv    per-clip WER values (same)
    study2_report.md                 aggregate tables only, no file names
    study2_results.json              the same numbers, machine readable
    study2_agreement_by_measure.png  agreement by measure (the paper's Figure 4)

    python3 tools/pipeline/study2_reliability.py --pass-a A.xlsx --pass-b B.xlsx
    python3 tools/pipeline/study2_reliability.py --pass-a A.xlsx --pass-b B.xlsx --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd

import common
import recipe
from validate import agreement_stats

SHEET_TRANSCRIPTS = "Transcriptions"
SHEET_RECALL = "Score - word recall"
RECALL_SCORE_COLS = ["Correct", "Repetitions", "Intrusions", "Primacy", "Middle", "Recency"]
REQUIRED_COLS = {
    SHEET_TRANSCRIPTS: ["clip_id", "audio_file", "task", "transcription"],
    SHEET_RECALL: ["clip_id", "audio_file"] + RECALL_SCORE_COLS,
}
TASKS = ["word_recall"]
TASK_LABEL = {"word_recall": "word-list recall"}

# One entry per reported measure: key, label, human sheet, human column,
# machine task, machine column candidates (the first one present wins).
# Primacy/middle/recency use the 3-6-3 split the RAs score (positions 1-3,
# 4-9, 10-12), which the serial table stores as *_3_6_3.
MEASURES = [
    ("recall_correct", "Recall correct", SHEET_RECALL, "Correct", "word_recall", ["n_correct"]),
    ("recall_repetitions", "Recall repetitions", SHEET_RECALL, "Repetitions", "word_recall",
     ["n_repetitions"]),
    ("recall_intrusions", "Recall intrusions", SHEET_RECALL, "Intrusions", "word_recall",
     ["n_intrusions"]),
    ("recall_primacy", "Primacy", SHEET_RECALL, "Primacy", "word_recall",
     ["primacy", "primacy_3_6_3", "n_primacy"]),
    ("recall_middle", "Middle", SHEET_RECALL, "Middle", "word_recall",
     ["middle", "middle_3_6_3", "n_middle"]),
    ("recall_recency", "Recency", SHEET_RECALL, "Recency", "word_recall",
     ["recency", "recency_3_6_3", "n_recency"]),
]
# comparison key -> (label, reference column, comparison column). Bias is
# comparison minus reference, so for machine rows it is machine minus human
# and for the human ceiling it is pass B minus pass A.
COMPARISONS = {
    "machine_vs_passA": ("machine vs pass A", "passA", "machine"),
    "machine_vs_passB": ("machine vs pass B", "passB", "machine"),
    "passA_vs_passB": ("pass A vs pass B", "passA", "passB"),
    "machine_vs_human_mean": ("machine vs mean of A and B", "human_mean", "machine"),
}
STAT_COLS = ["n", "icc21", "mae", "bias", "pearson_r", "exact_match_rate", "within1_rate"]
WER_COLS = ["n", "mean_wer", "median_wer", "corpus_wer"]

TEAL = "#1B7F73"    # machine vs human
AMBER = "#C08A2A"   # human vs human


# --- small helpers ----------------------------------------------------------
def _is_blank(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) \
        or str(value).strip() == ""


def clean_basename(value) -> str:
    """Join key: the audio file name with every whitespace character removed."""
    return "" if _is_blank(value) else "".join(str(value).split())


def canon_task(value) -> str | None:
    """Map a workbook task label ('word-list recall', 'letter fluency', ...) to
    the pipeline task name; None when the cell says something else."""
    s = "" if _is_blank(value) else str(value).lower()
    if "recall" in s:
        return "word_recall"
    if "letter" in s:
        return "letter_fluency"
    if "category" in s:
        return "category_fluency"
    return None


def coerce_numeric(series: pd.Series) -> tuple[pd.Series, int, int]:
    """Turn a hand-scored column into floats without raising.

    Blank cells become NaN and are counted; cells holding something other than
    a number (a '?', a word, two numbers) also become NaN and are counted
    separately so the report can say how many cells were unusable.
    """
    text = series.map(lambda v: "" if _is_blank(v) else str(v).strip())
    blank = text == ""
    values = pd.to_numeric(text.where(~blank, other=np.nan), errors="coerce")
    nonnumeric = (~blank) & values.isna()
    return values.astype(float), int(blank.sum()), int(nonnumeric.sum())


def _nonblank_any(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """True for rows where at least one of `cols` holds something."""
    if not cols or df.empty:
        return pd.Series(False, index=df.index)
    return df[cols].apply(lambda s: s.map(lambda v: not _is_blank(v))).any(axis=1)


def _basenames(df: pd.DataFrame, clip_to_base: dict[str, str]) -> pd.Series:
    """Basename per row; a blank audio_file falls back to the clip_id lookup."""
    base = df["audio_file"].map(clean_basename)
    fallback = df["clip_id"].map(lambda v: clip_to_base.get(str(v).strip(), ""))
    return base.where(base != "", fallback)


def _fmt(value, nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{nd}f}"


def _jsonable(obj):
    """Recursively convert numpy scalars and NaN so json.dumps accepts them."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    return obj


# --- workbooks --------------------------------------------------------------
@dataclass
class PassData:
    """One completed workbook, keyed by basename."""
    label: str
    transcripts: pd.DataFrame   # basename, task, transcription
    recall: pd.DataFrame        # basename + RECALL_SCORE_COLS (float)
    quality: dict = field(default_factory=dict)


def read_sheet(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", dtype=object)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def check_workbook(path: str, label: str) -> list[str]:
    """Problems that make a workbook unusable (missing file, sheet or column)."""
    if not os.path.exists(path):
        return [f"{label}: file not found: {path}"]
    try:
        sheets = pd.ExcelFile(path, engine="openpyxl").sheet_names
    except Exception as exc:  # noqa: BLE001  (a corrupt workbook is a report line, not a traceback)
        return [f"{label}: cannot open workbook ({exc})"]
    problems = []
    for sheet, cols in REQUIRED_COLS.items():
        if sheet not in sheets:
            problems.append(f"{label}: sheet '{sheet}' missing")
            continue
        have = set(read_sheet(path, sheet).columns)
        problems += [f"{label}: sheet '{sheet}' lacks column '{c}'" for c in cols if c not in have]
    return problems


def load_pass(path: str, label: str) -> PassData:
    """Read one completed workbook into basename-keyed frames, counting what
    is unusable (blank, non-numeric, unscored, duplicate, unknown task). Only
    recall rows are kept; other tasks in the Transcriptions tab are ignored."""
    q = {"rows": {}, "unscored_rows": {}, "duplicates": {}, "blank": {},
         "nonnumeric": {}, "unknown_task": {}}

    tr = read_sheet(path, SHEET_TRANSCRIPTS)
    tr["basename"] = tr["audio_file"].map(clean_basename)
    tr = tr[tr["basename"] != ""].copy()
    clip_to_base = {str(c).strip(): b for c, b in zip(tr["clip_id"], tr["basename"])}
    tr["task"] = tr["task"].map(canon_task)
    q["unknown_task"][SHEET_TRANSCRIPTS] = int(tr["task"].isna().sum())
    tr = tr[tr["task"] == "word_recall"].copy()
    q["rows"][SHEET_TRANSCRIPTS] = len(tr)
    tr["transcription"] = tr["transcription"].map(lambda v: "" if _is_blank(v) else str(v).strip())
    q["unscored_rows"][SHEET_TRANSCRIPTS] = int((tr["transcription"] == "").sum())
    tr = tr[(tr["transcription"] != "") & tr["task"].notna()]
    q["duplicates"][SHEET_TRANSCRIPTS] = int(tr["basename"].duplicated().sum())
    tr = tr.drop_duplicates("basename", keep="first")
    tr = tr[["basename", "task", "transcription"]].reset_index(drop=True)

    rc = read_sheet(path, SHEET_RECALL)
    rc["basename"] = _basenames(rc, clip_to_base)
    rc = rc[rc["basename"] != ""].copy()
    q["rows"][SHEET_RECALL] = len(rc)
    scored = _nonblank_any(rc, RECALL_SCORE_COLS)
    q["unscored_rows"][SHEET_RECALL] = int((~scored).sum())
    rc = rc[scored].copy()
    for col in RECALL_SCORE_COLS:
        rc[col], nb, nn = coerce_numeric(rc[col])
        q["blank"][f"{SHEET_RECALL} / {col}"] = nb
        q["nonnumeric"][f"{SHEET_RECALL} / {col}"] = nn
    q["duplicates"][SHEET_RECALL] = int(rc["basename"].duplicated().sum())
    rc = rc.drop_duplicates("basename", keep="first")
    rc = rc[["basename"] + RECALL_SCORE_COLS].reset_index(drop=True)
    return PassData(label, tr, rc, q)



def human_series(p: PassData, sheet: str, col: str, task: str) -> pd.Series:
    """The hand-scored values for one measure, indexed by basename."""
    return p.recall.set_index("basename")[col].astype(float)


# --- machine side -----------------------------------------------------------
@dataclass
class MachineData:
    scores: dict[str, pd.Series]        # measure key -> values indexed by basename
    transcripts: dict[str, str]         # basename -> production transcript text
    notes: list[str]
    recipe: dict
    transcripts_missing: int = 0


def _read_by_basename(path: str | None) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    if "basename" not in df.columns:
        return None
    df["basename"] = df["basename"].map(clean_basename)
    return df.drop_duplicates("basename", keep="first").set_index("basename")


def load_machine(args, basenames_needed: list[str]) -> MachineData:
    """Machine scores per measure plus the production transcripts for the
    clips the workbooks cover."""
    notes, scores, recipe = [], {}, {}
    recall = _read_by_basename(args.scores_recall)
    if recall is None:
        notes.append(f"recall score table unusable: {args.scores_recall}")
    else:
        serial = _read_by_basename(args.scores_serial)
        if serial is None:
            notes.append(f"serial-position table not found: {args.scores_serial} "
                         "(primacy/middle/recency need it unless the recall table has them)")
        else:
            extra = [c for c in serial.columns if c not in recall.columns]
            recall = recall.join(serial[extra], how="left")
        for col in ("recipe_version", "recipe_config"):
            if col in recall.columns:
                recipe[col] = sorted(recall[col].dropna().astype(str).unique().tolist())
    for key, _label, _sheet, _hcol, task, candidates in MEASURES:
        frame = recall
        if frame is None:
            scores[key] = pd.Series(dtype=float)
            continue
        col = next((c for c in candidates if c in frame.columns), None)
        if col is None:
            notes.append(f"{key}: none of {candidates} in the machine table; "
                         "machine side of this measure is empty")
            scores[key] = pd.Series(dtype=float)
        else:
            scores[key] = pd.to_numeric(frame[col], errors="coerce").astype(float)

    transcripts, missing = {}, 0
    for b in basenames_needed:
        p = os.path.join(args.transcripts_dir, b + ".json")
        if not os.path.exists(p):
            missing += 1
            continue
        try:
            with open(p) as fh:
                transcripts[b] = str(json.load(fh).get("text") or "")
        except (OSError, ValueError):
            missing += 1
    return MachineData(scores, transcripts, notes, recipe, missing)


# --- agreement --------------------------------------------------------------
def score_agreement(pass_a: PassData, pass_b: PassData, machine: MachineData
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per-measure agreement rows, the per-clip pairs behind them, and the
    matching counts (clips scored by either pass, with or without a machine
    row, present in one pass only)."""
    rows, paired, matching = [], [], {}
    for key, label, sheet, hcol, task, _cands in MEASURES:
        ha = human_series(pass_a, sheet, hcol, task)
        hb = human_series(pass_b, sheet, hcol, task)
        m = machine.scores.get(key, pd.Series(dtype=float)).rename("machine")
        df = pd.DataFrame({"passA": ha, "passB": hb}).join(m, how="outer")
        df = df[df["passA"].notna() | df["passB"].notna()].copy()
        df["human_mean"] = df[["passA", "passB"]].mean(axis=1, skipna=False)
        matching[key] = {
            "clips_scored_by_either_pass": len(df),
            "with_machine_row": int(df["machine"].notna().sum()),
            "without_machine_row": int(df["machine"].isna().sum()),
            "passA_only": int((df["passA"].notna() & df["passB"].isna()).sum()),
            "passB_only": int((df["passB"].notna() & df["passA"].isna()).sum()),
        }
        for ckey, (clabel, ref, cmp_) in COMPARISONS.items():
            st = agreement_stats(df[ref], df[cmp_])
            rows.append({"measure": key, "measure_label": label,
                         "comparison": ckey, "comparison_label": clabel,
                         "n": int(st.get("n", 0)), "icc21": st.get("icc21", np.nan),
                         "mae": st.get("mae", np.nan),
                         "bias": st.get("bias_auto_minus_ref", np.nan),
                         "pearson_r": st.get("pearson_r", np.nan),
                         "exact_match_rate": st.get("exact_match_rate", np.nan),
                         "within1_rate": st.get("within1_rate", np.nan)})
        paired.append(df.rename_axis("basename").reset_index().assign(measure=key))
    table = pd.DataFrame(rows)
    pairs = pd.concat(paired, ignore_index=True) if paired else pd.DataFrame()
    return table, pairs, matching


def _wer_pair(reference: str | None, hypothesis: str | None):
    """(wer, n_ref, edits) or None when the pair cannot be scored."""
    if reference is None or hypothesis is None or not str(reference).strip():
        return None
    w, n_ref = common.wer(reference, hypothesis)
    if n_ref == 0 or (isinstance(w, float) and np.isnan(w)):
        return None
    return float(w), int(n_ref), float(w) * n_ref


def transcript_agreement(pass_a: PassData, pass_b: PassData, machine: MachineData
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """WER per task for pass A vs pass B (reference = A), machine vs each pass
    (reference = the human pass) and machine vs the mean of the two."""
    a = pass_a.transcripts.set_index("basename")
    b = pass_b.transcripts.set_index("basename")
    df = a[["task", "transcription"]].rename(columns={"transcription": "passA"}).join(
        b[["task", "transcription"]].rename(columns={"task": "task_b", "transcription": "passB"}),
        how="outer")
    df["task"] = df["task"].where(df["task"].notna(), df["task_b"])
    df = df.drop(columns="task_b")
    df["machine"] = [machine.transcripts.get(k) for k in df.index]

    clip_rows = []
    for basename, r in df.iterrows():
        rec = {"basename": basename, "task": r["task"]}
        ab = _wer_pair(r["passA"], r["passB"])
        ma = _wer_pair(r["passA"], r["machine"])
        mb = _wer_pair(r["passB"], r["machine"])
        for ckey, val in (("passA_vs_passB", ab), ("machine_vs_passA", ma),
                          ("machine_vs_passB", mb)):
            rec[f"wer_{ckey}"] = val[0] if val else np.nan
            rec[f"nref_{ckey}"] = val[1] if val else np.nan
            rec[f"edits_{ckey}"] = val[2] if val else np.nan
        if ma and mb:
            rec["wer_machine_vs_human_mean"] = (ma[0] + mb[0]) / 2
            rec["nref_machine_vs_human_mean"] = ma[1] + mb[1]
            rec["edits_machine_vs_human_mean"] = ma[2] + mb[2]
        else:
            for c in ("wer", "nref", "edits"):
                rec[f"{c}_machine_vs_human_mean"] = np.nan
        clip_rows.append(rec)
    clip_cols = ["basename", "task"] + [f"{c}_{k}" for k in COMPARISONS
                                        for c in ("wer", "nref", "edits")]
    clips = pd.DataFrame(clip_rows, columns=clip_cols)  # columns survive an empty run

    rows = []
    for task in TASKS:
        sub = clips[clips["task"] == task]
        for ckey, (clabel, _ref, _cmp) in COMPARISONS.items():
            w = sub[f"wer_{ckey}"] if len(sub) else pd.Series(dtype=float)
            ok = w.notna()
            n = int(ok.sum())
            nref = sub.loc[ok, f"nref_{ckey}"].sum() if n else 0
            edits = sub.loc[ok, f"edits_{ckey}"].sum() if n else 0
            rows.append({"task": task, "task_label": TASK_LABEL[task],
                         "comparison": ckey, "comparison_label": clabel, "n": n,
                         "mean_wer": float(w[ok].mean()) if n else np.nan,
                         "median_wer": float(w[ok].median()) if n else np.nan,
                         "corpus_wer": float(edits / nref) if nref else np.nan})
    return pd.DataFrame(rows), clips


# --- figure -----------------------------------------------------------------
def make_figure(table: pd.DataFrame, path: str) -> None:
    """Paper Figure 5: ICC(2,1) per measure, machine vs human (teal, mean of
    the two passes with a whisker spanning them) beside human vs human (amber).
    Light background, direct labels, y from 0 to 1, reference line at 0.90."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2B2B2B", "#6B6B6B", "#E3E3E3"
    labels, mh, lo, hi, hh = [], [], [], [], []
    for key, label, *_ in MEASURES:
        t = table[table["measure"] == key].set_index("comparison")["icc21"]
        pair = [v for v in (t.get("machine_vs_passA", np.nan), t.get("machine_vs_passB", np.nan))
                if pd.notna(v)]
        labels.append(label.replace(" ", "\n", 1))
        mh.append(float(np.mean(pair)) if pair else np.nan)
        lo.append(min(pair) if pair else np.nan)
        hi.append(max(pair) if pair else np.nan)
        hh.append(float(t.get("passA_vs_passB", np.nan)))

    x = np.arange(len(labels))
    width, gap = 0.26, 0.02
    xm, xh = x - (width + gap) / 2, x + (width + gap) / 2
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def heights(vals):
        return [max(v, 0.0) if pd.notna(v) else 0.0 for v in vals]

    ax.bar(xm, heights(mh), width=width, color=TEAL, zorder=3,
           label="Machine vs human (mean of pass A and pass B)")
    ax.bar(xh, heights(hh), width=width, color=AMBER, zorder=3,
           label="Human vs human (pass A vs pass B)")
    for xi, l, h in zip(xm, lo, hi):
        if pd.notna(l) and pd.notna(h) and h > l:
            ax.plot([xi, xi], [max(l, 0.0), min(h, 1.0)], color=ink, lw=1.0,
                    solid_capstyle="butt", zorder=4)  # whisker stays inside the axis
    for xi, v, top in list(zip(xm, mh, hi)) + list(zip(xh, hh, hh)):
        if pd.isna(v):
            ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=7.5, color=muted)
        else:
            y = min(max(v, 0.0, top if pd.notna(top) else 0.0), 1.0)
            ax.text(xi, y + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=ink)

    ax.axhline(0.90, color=muted, lw=0.8, zorder=2)
    ax.text(x[-1] + 0.5, 0.905, "0.90", ha="right", va="bottom", fontsize=7.5, color=muted)
    ax.set_xlim(-0.6, x[-1] + 0.6)
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.spines["left"].set_bounds(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color=ink)
    ax.set_ylabel("ICC(2,1)", color=ink, fontsize=9)
    ax.yaxis.grid(True, color=grid, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink, length=0, labelsize=8)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8, ncol=2,
              handlelength=1.2, columnspacing=1.6)
    ax.set_title("Study 2: agreement by measure", loc="left", pad=26, fontsize=11, color=ink)
    fig.text(0.01, 0.01, "Whisker on the teal bar spans machine vs pass A and machine vs pass B. "
             "Grey line marks ICC 0.90.", fontsize=7.5, color=muted, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# --- report -----------------------------------------------------------------
def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _quality_rows(qa: dict, qb: dict) -> list[list[str]]:
    rows = []
    for sheet in (SHEET_TRANSCRIPTS, SHEET_RECALL):
        rows.append([f"{sheet}: rows with an audio file", str(qa["rows"].get(sheet, 0)),
                     str(qb["rows"].get(sheet, 0))])
        what = "untranscribed" if sheet == SHEET_TRANSCRIPTS else "not yet scored (all cells blank)"
        rows.append([f"{sheet}: rows {what}", str(qa["unscored_rows"].get(sheet, 0)),
                     str(qb["unscored_rows"].get(sheet, 0))])
        rows.append([f"{sheet}: duplicate audio files", str(qa["duplicates"].get(sheet, 0)),
                     str(qb["duplicates"].get(sheet, 0))])
        if sheet in qa["unknown_task"]:
            rows.append([f"{sheet}: unrecognised task label", str(qa["unknown_task"].get(sheet, 0)),
                         str(qb["unknown_task"].get(sheet, 0))])
    for key in qa["blank"]:
        rows.append([f"{key}: blank cells in scored rows", str(qa["blank"][key]),
                     str(qb["blank"].get(key, 0))])
    for key in qa["nonnumeric"]:
        rows.append([f"{key}: non-numeric cells", str(qa["nonnumeric"][key]),
                     str(qb["nonnumeric"].get(key, 0))])
    return rows


def exploratory_versions(machine: MachineData) -> list[str]:
    """Recipe versions in the machine table other than the frozen one; a
    non-empty list means the run is exploratory and must be marked."""
    return [v for v in machine.recipe.get("recipe_version", []) if v != recipe.FROZEN_VERSION]


def exploratory_banner(versions: list[str]) -> str:
    return (f"EXPLORATORY: machine scores come from recipe version {', '.join(versions)}, "
            f"not the frozen recipe {recipe.FROZEN_VERSION}. These numbers are not the "
            "pre-specified Study 2 result; report outputs/study2/ for that.")


def write_report(path: str, args, pass_a: PassData, pass_b: PassData, machine: MachineData,
                 scores: pd.DataFrame, wer: pd.DataFrame, matching: dict, n_clips: int) -> None:
    """Markdown report with aggregate tables only (no file names, no per-clip rows)."""
    recipe_str = ", ".join(f"{k} {'/'.join(v)}" for k, v in machine.recipe.items()) or "not recorded"
    today = dt.datetime.now(tz=dt.timezone.utc).astimezone().date().isoformat()
    sources = f"`{os.path.basename(args.scores_recall)}`"
    found = n_clips - machine.transcripts_missing
    exploratory = exploratory_versions(machine)
    lines = ["# Study 2: reliability against two blind human passes", ""]
    if exploratory:
        lines += [f"**{exploratory_banner(exploratory)}**", ""]
    lines += [
        f"Generated {today}. Machine scores: {sources}; recipe {recipe_str}.", "",
        "Aggregate tables only. Per-clip pairs are written to CSVs beside the report in outputs/, "
        "which is not tracked.", "",
        "## Inputs and data quality", "",
        _md_table(["Item", "Pass A", "Pass B"], _quality_rows(pass_a.quality, pass_b.quality)), "",
        f"Machine transcript found for {found} of {n_clips} clips transcribed by at least one pass.",
        "",
    ]
    if machine.notes:
        lines += ["Machine-side notes:", ""] + [f"- {n}" for n in machine.notes] + [""]
    matching_cols = ("clips_scored_by_either_pass", "with_machine_row", "without_machine_row",
                     "passA_only", "passB_only")
    matching_rows = [[label] + [str(matching[k][c]) for c in matching_cols]
                     for k, label, *_ in MEASURES]
    score_rows = [[r["measure_label"], r["comparison_label"], str(r["n"]), _fmt(r["icc21"]),
                   _fmt(r["mae"], 2), _fmt(r["bias"], 2), _fmt(r["exact_match_rate"], 2),
                   _fmt(r["within1_rate"], 2)] for r in scores.to_dict("records")]
    wer_rows = [[r["task_label"], r["comparison_label"], str(r["n"]), _fmt(r["mean_wer"]),
                 _fmt(r["median_wer"]), _fmt(r["corpus_wer"])] for r in wer.to_dict("records")]
    icc_note = ("ICC(2,1) is two-way random effects, single rater, absolute agreement "
                "(validate.agreement_stats). Bias is the second member of the comparison minus "
                "the first: machine minus human, or pass B minus pass A.")
    wer_note = ("WER is edit distance over reference words (common.wer). The reference is the "
                "human pass named in the comparison (pass A for the human-human row); the "
                "mean-of-humans row averages the two machine WERs per clip. Corpus WER pools "
                "edits and reference words across clips.")
    fig_note = ("`study2_agreement_by_measure.png`: ICC(2,1) per measure, machine vs "
                "human in teal (mean of the two passes, whisker spanning them) and human vs "
                "human in amber, reference line at 0.90.")
    lines += [
        "## Matching", "",
        _md_table(["Measure", "Scored by either pass", "With machine row", "Without machine row",
                   "Pass A only", "Pass B only"], matching_rows), "",
        "## Score agreement", "", icc_note, "",
        _md_table(["Measure", "Comparison", "n", "ICC(2,1)", "MAE", "Bias", "Exact", "Within 1"],
                  score_rows), "",
        "## Transcript agreement (word error rate)", "", wer_note, "",
        _md_table(["Task", "Comparison", "n", "Mean WER", "Median WER", "Corpus WER"], wer_rows), "",
        "## Figure", "", fig_note, "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


# --- driver -----------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pass-a", required=True, help="completed pass A workbook (.xlsx)")
    ap.add_argument("--pass-b", required=True, help="completed pass B workbook (.xlsx)")
    ap.add_argument("--scores-recall", default=os.path.join(_REPO, "outputs", "scores_word_recall.csv"))
    ap.add_argument("--scores-serial", default=None,
                    help="serial-position table for primacy/middle/recency "
                         "(default: scores_serial.csv beside --scores-recall)")
    ap.add_argument("--transcripts-dir", default=os.path.join(_REPO, "cache", "transcripts"))
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "outputs", "study2"))
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the inputs and print what would be computed; write nothing")
    args = ap.parse_args(argv)
    if args.scores_serial is None:
        args.scores_serial = os.path.join(os.path.dirname(os.path.abspath(args.scores_recall)),
                                          "scores_serial.csv")
    return args


def _print_quality(p: PassData) -> None:
    q = p.quality
    for sheet in (SHEET_TRANSCRIPTS, SHEET_RECALL):
        print(f"[study2] {p.label} {sheet}: {q['rows'].get(sheet, 0)} rows, "
              f"{q['unscored_rows'].get(sheet, 0)} not filled in, "
              f"{q['duplicates'].get(sheet, 0)} duplicate files, "
              f"{q['unknown_task'].get(sheet, 0)} unknown task")
    blanks = {k: v for k, v in q["blank"].items() if v}
    nonnum = {k: v for k, v in q["nonnumeric"].items() if v}
    if blanks:
        print(f"[study2] {p.label} blank cells in scored rows: {blanks}")
    if nonnum:
        print(f"[study2] {p.label} non-numeric cells (dropped): {nonnum}")


def main(argv=None) -> int:
    args = parse_args(argv)
    problems = check_workbook(args.pass_a, "pass A") + check_workbook(args.pass_b, "pass B")
    for path, name in ((args.scores_recall, "recall score table"),):
        if not os.path.exists(path):
            problems.append(f"{name} not found: {path}")
    if not os.path.isdir(args.transcripts_dir):
        problems.append(f"transcripts directory not found: {args.transcripts_dir}")
    if problems:
        for p in problems:
            print(f"[study2] PROBLEM: {p}")
        return 2

    pass_a = load_pass(args.pass_a, "pass A")
    pass_b = load_pass(args.pass_b, "pass B")
    needed = sorted(set(pass_a.transcripts["basename"]) | set(pass_b.transcripts["basename"]))
    machine = load_machine(args, needed)
    _print_quality(pass_a)
    _print_quality(pass_b)
    print(f"[study2] machine transcripts: {len(machine.transcripts)} found, "
          f"{machine.transcripts_missing} missing of {len(needed)} workbook clips")
    for note in machine.notes:
        print(f"[study2] note: {note}")

    scores, paired_scores, matching = score_agreement(pass_a, pass_b, machine)
    wer, paired_wer = transcript_agreement(pass_a, pass_b, machine)

    outputs = {
        "scores": os.path.join(args.out_dir, "study2_score_agreement.csv"),
        "wer": os.path.join(args.out_dir, "study2_transcript_agreement.csv"),
        "paired_scores": os.path.join(args.out_dir, "study2_paired_scores.csv"),
        "paired_wer": os.path.join(args.out_dir, "study2_paired_transcripts.csv"),
        "report": os.path.join(args.out_dir, "study2_report.md"),
        "json": os.path.join(args.out_dir, "study2_results.json"),
        "figure": os.path.join(args.out_dir, "study2_agreement_by_measure.png"),
    }
    if args.dry_run:
        print("[study2] dry run: inputs are valid; nothing written. Would compute:")
        for k, label, *_ in MEASURES:
            ns = {c: int(scores[(scores.measure == k) & (scores.comparison == c)]["n"].iloc[0])
                  for c in COMPARISONS}
            print(f"[study2]   {label}: " + ", ".join(f"{c} n={n}" for c, n in ns.items()))
        for task in TASKS:
            ns = {c: int(wer[(wer.task == task) & (wer.comparison == c)]["n"].iloc[0])
                  for c in COMPARISONS}
            print(f"[study2]   WER {TASK_LABEL[task]}: " + ", ".join(f"{c} n={n}" for c, n in ns.items()))
        print("[study2] outputs that would be written:")
        for p in outputs.values():
            print(f"[study2]   {p}")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    exploratory = exploratory_versions(machine)
    if exploratory:
        outputs["readme"] = os.path.join(args.out_dir, recipe.README_NAME)
        recipe.write_exploratory_readme(args.out_dir, exploratory[0],
                                        extra="Study 2 agreement tables in this directory: "
                                              + exploratory_banner(exploratory))
        print(f"[study2] {exploratory_banner(exploratory)}")
    scores.to_csv(outputs["scores"], index=False)
    wer.to_csv(outputs["wer"], index=False)
    paired_scores.to_csv(outputs["paired_scores"], index=False)
    paired_wer.to_csv(outputs["paired_wer"], index=False)
    make_figure(scores, outputs["figure"])
    write_report(outputs["report"], args, pass_a, pass_b, machine, scores, wer, matching, len(needed))
    results = {
        "generated": dt.datetime.now(tz=dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "inputs": {k: os.path.basename(os.path.normpath(v)) for k, v in (
            ("pass_a", args.pass_a), ("pass_b", args.pass_b), ("scores_recall", args.scores_recall),
            ("scores_serial", args.scores_serial), ("transcripts_dir", args.transcripts_dir))},
        "recipe": machine.recipe,
        "exploratory": bool(exploratory),
        "exploratory_note": exploratory_banner(exploratory) if exploratory else None,
        "data_quality": {"passA": pass_a.quality, "passB": pass_b.quality,
                         "machine_notes": machine.notes,
                         "workbook_clips_transcribed": len(needed),
                         "machine_transcripts_missing": machine.transcripts_missing},
        "matching": matching,
        "score_agreement": scores.to_dict("records"),
        "transcript_agreement": wer.to_dict("records"),
    }
    with open(outputs["json"], "w") as fh:
        json.dump(_jsonable(results), fh, indent=2)

    show = scores[["measure_label", "comparison_label"] + STAT_COLS[:4]]
    print(show.to_string(index=False))
    print(wer[["task_label", "comparison_label"] + WER_COLS].to_string(index=False))
    print(f"[study2] wrote {len(outputs)} files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
