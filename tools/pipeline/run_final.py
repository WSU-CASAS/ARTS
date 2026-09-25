#!/usr/bin/env python3
"""Study 2 driver: run the frozen recipe end to end on the held-out set.

Steps, in order (each one is a plain command so it can also be run by hand):

  1. index          rebuild outputs/trial_index.csv and the keys (only when the
                    index is missing or --rebuild-index is given)
  2. transcribe     Whisper medium for every indexed recording that is not yet
                    in cache/transcripts/ (a no-op when everything is cached)
  3. words          the word-level pass into cache/transcripts_words/ (same rule)
  4. alternatives   the teacher-forced alternatives pass for the target
                    recordings (recall recordings of the locked participants)
                    that the alternatives tables do not cover yet
  5. score          score_wordrecall, score_serial, export_token_classification,
                    all under the frozen recipe; each writes outputs/recipe_manifest.json
  6. guard          validate_recipe.py --check (development-set numbers must
                    not have moved)
  7. study2         study2_reliability.py against the two completed workbooks

The driver refuses to run when outputs/recipe_manifest.json was written by a
different recipe version than recipe.RECIPE_VERSION: the score tables on disk
would then belong to another recipe, and the manifest must be regenerated
deliberately (delete it and re-run) rather than silently overwritten.

    python3 tools/pipeline/run_final.py --dry-run
    python3 tools/pipeline/run_final.py --pass-a A.xlsx --pass-b B.xlsx
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

import pandas as pd

import config
import recipe

PY = "python3"
WHISPER_MODEL = "medium"  # the validated caches were built with this model
LOCKED_PARTICIPANTS = config.LOCKED_PARTICIPANTS   # keys/locked_participants.txt (not tracked)
WORKBOOK_DIR = config.PROJECT_ROOT
DEFAULT_PASS_A = WORKBOOK_DIR / "CAT_reliability20_passA.xlsx"
DEFAULT_PASS_B = WORKBOOK_DIR / "CAT_reliability20_passB.xlsx"
WORK_DIR = config.OUT_DIR / "run_final"


@dataclass
class Step:
    name: str
    commands: list[list[str]]
    env: dict = field(default_factory=dict)
    skip_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def render(self) -> list[str]:
        out = []
        for cmd in self.commands:
            prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in self.env.items())
            out.append(prefix + " ".join(shlex.quote(c) for c in cmd))
        return out


def _rel(p) -> str:
    try:
        return os.path.relpath(str(p), _REPO)
    except ValueError:
        return str(p)


def _write_list(name: str, items: list[str]) -> str:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    p = WORK_DIR / name
    p.write_text("\n".join(items) + ("\n" if items else ""))
    return _rel(p)


def check_inputs() -> list[str]:
    """Problems that stop a run before anything is executed."""
    problems = []
    for label, p in [("audio root", config.AUDIO_ROOT), ("prompts zip", config.PROMPTS_ZIP),
                     ("word lists", config.WORD_LISTS_JSON),
                     ("presentation order", config.WORD_LISTS_ORDERED_JSON)]:
        if not p.exists():
            problems.append(f"{label} not found: {p}")
    if not LOCKED_PARTICIPANTS:
        problems.append("no held-out participants listed: create keys/locked_participants.txt "
                        "or set SWC_LOCKED_PARTICIPANTS (see config.py)")
    return problems


def check_manifest() -> str | None:
    """Refuse when the manifest on disk belongs to another recipe version."""
    m = recipe.read_manifest()
    if m is None:
        return None
    if m.get("recipe_version") != recipe.RECIPE_VERSION:
        return (f"outputs/recipe_manifest.json was written by recipe "
                f"{m.get('recipe_version')!r}, this code is {recipe.RECIPE_VERSION!r}; "
                "delete the manifest (and the score tables it describes) to re-score deliberately")
    return None


def plan(args) -> list[Step]:
    steps = []
    index_exists = config.INDEX_CSV.exists()
    s = Step("index", [[PY, "src/io_index.py"]])
    if index_exists and not args.rebuild_index:
        s.skip_reason = f"{_rel(config.INDEX_CSV)} exists (pass --rebuild-index to redo)"
    steps.append(s)

    idx = pd.read_csv(config.INDEX_CSV) if index_exists else pd.DataFrame()
    idx = idx[idx["audio_found"]] if len(idx) else idx
    basenames = list(idx["basename"]) if len(idx) else []

    missing = [b for b in basenames if not (config.TRANSCRIPT_CACHE / f"{b}.json").exists()]
    s = Step("transcribe", [])
    if not index_exists:
        s.notes.append("index missing: the list of uncached recordings is computed after step 1")
        s.commands = [[PY, "src/transcribe.py", "--model", WHISPER_MODEL]]
    elif missing:
        lst = _write_list("missing_transcripts.txt", missing) if not args.dry_run \
            else _rel(WORK_DIR / "missing_transcripts.txt")
        s.commands = [[PY, "src/transcribe.py", "--model", WHISPER_MODEL, "--only", lst]]
        s.notes.append(f"{len(missing)} of {len(basenames)} recordings not in {_rel(config.TRANSCRIPT_CACHE)}")
    else:
        s.skip_reason = f"all {len(basenames)} recordings cached in {_rel(config.TRANSCRIPT_CACHE)}"
    steps.append(s)

    missing_w = [b for b in basenames if not (config.WORDS_CACHE / f"{b}.json").exists()]
    s = Step("words", [], env={"SWC_TRANSCRIPT_CACHE": _rel(config.WORDS_CACHE)})
    if not index_exists:
        s.commands = [[PY, "src/transcribe.py", "--model", WHISPER_MODEL, "--word-timestamps"]]
    elif missing_w:
        lst = _write_list("missing_word_transcripts.txt", missing_w) if not args.dry_run \
            else _rel(WORK_DIR / "missing_word_transcripts.txt")
        s.commands = [[PY, "src/transcribe.py", "--model", WHISPER_MODEL, "--word-timestamps",
                       "--only", lst]]
        s.notes.append(f"{len(missing_w)} of {len(basenames)} recordings not in {_rel(config.WORDS_CACHE)}")
    else:
        s.skip_reason = f"all {len(basenames)} recordings cached in {_rel(config.WORDS_CACHE)}"
    steps.append(s)

    s = Step("alternatives", [])
    if len(idx):
        rec = idx[(idx["task"] == "word_recall") & idx["participant_num"].isin(LOCKED_PARTICIPANTS)]
        covered = set()
        for p in config.ALTERNATIVES_CSVS:
            if p.exists():
                covered |= set(pd.read_csv(p, usecols=["basename"])["basename"])
        targets = [b for b in rec["basename"] if b not in covered]
        out_csv = config.ALTERNATIVES_CSVS[-1]
        if targets:
            lst = _write_list("alternatives_targets.txt", targets) if not args.dry_run \
                else _rel(WORK_DIR / "alternatives_targets.txt")
            s.commands = [[PY, "tools/pipeline/word_alternatives.py", "--targets", lst,
                           "--model", WHISPER_MODEL, "--out", _rel(out_csv)]]
            s.notes.append(f"{len(targets)} of {len(rec)} held-out recall recordings lack "
                           f"alternatives rows (R3 rescue needs them)")
        else:
            s.skip_reason = f"all {len(rec)} held-out recall recordings have alternatives rows"
    else:
        s.notes.append("index missing: targets are computed after step 1")
        s.commands = [[PY, "tools/pipeline/word_alternatives.py", "--targets",
                       _rel(WORK_DIR / "alternatives_targets.txt"), "--model", WHISPER_MODEL,
                       "--out", _rel(config.ALTERNATIVES_CSVS[-1])]]
    steps.append(s)

    steps.append(Step("score", [
        [PY, "src/score_wordrecall.py"],
        [PY, "src/score_serial.py"],
        [PY, "tools/pipeline/export_token_classification.py"],
    ], env={"SWC_RECIPE_CONFIG": config.RECIPE_CONFIG}))
    steps.append(Step("guard", [[PY, "tools/pipeline/validate_recipe.py", "--check"]]))

    s = Step("study2", [[PY, "tools/pipeline/study2_reliability.py",
                         "--pass-a", str(args.pass_a), "--pass-b", str(args.pass_b)]])
    for label, p in (("pass A", args.pass_a), ("pass B", args.pass_b)):
        if not os.path.exists(p):
            s.notes.append(f"{label} workbook not found: {p} (completed copies pending)")
    steps.append(s)
    return steps


def run_step(step: Step) -> int:
    env = {**os.environ, **step.env}
    for cmd in step.commands:
        print(f"[run_final] $ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=_REPO, env=env, check=False).returncode
        if rc != 0:
            print(f"[run_final] step {step.name} failed (exit {rc}); stopping")
            return rc
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="list the steps, check the inputs, print the commands; run nothing")
    ap.add_argument("--pass-a", default=str(DEFAULT_PASS_A), help="completed pass A workbook")
    ap.add_argument("--pass-b", default=str(DEFAULT_PASS_B), help="completed pass B workbook")
    ap.add_argument("--rebuild-index", action="store_true", help="re-run io_index even if the index exists")
    ap.add_argument("--from-step", default="index",
                    help="start at this step (index, transcribe, words, alternatives, score, guard, study2)")
    args = ap.parse_args(argv)

    print(f"[run_final] recipe {recipe.RECIPE_VERSION}, configuration {config.RECIPE_CONFIG}, "
          f"repo {_REPO}")
    refusal = check_manifest()
    problems = check_inputs()
    steps = plan(args)
    names = [s.name for s in steps]
    if args.from_step not in names:
        ap.error(f"--from-step must be one of {names}")
    start = names.index(args.from_step)

    for i, s in enumerate(steps, 1):
        state = "skip" if s.skip_reason else ("run" if i - 1 >= start else "before --from-step")
        print(f"\n[{i}] {s.name} ({state})")
        if s.skip_reason:
            print(f"    no-op: {s.skip_reason}")
        for n in s.notes:
            print(f"    note: {n}")
        for line in s.render():
            print(f"    $ {line}")
    print()
    for p in problems:
        print(f"[run_final] PROBLEM: {p}")
    if refusal:
        print(f"[run_final] REFUSING: {refusal}")
    if args.dry_run:
        print("[run_final] dry run: nothing executed")
        return 2 if (problems or refusal) else 0
    if refusal or problems:
        return 2
    for s in steps[start:]:
        if s.skip_reason:
            print(f"[run_final] {s.name}: skipped ({s.skip_reason})")
            continue
        if s.name == "study2" and any("not found" in n for n in s.notes):
            print("[run_final] study2: workbooks missing; scoring is done, run study2 later")
            return 0
        rc = run_step(s)
        if rc:
            return rc
    print("[run_final] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
