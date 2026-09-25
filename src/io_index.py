"""Build the master trial index and the scoring keys.

Outputs
-------
outputs/trial_index.csv : one row per recall recording, with the resolved audio
    path, the stimulus (target words) and the watch-reported counts where available.
keys/word_lists.json    : {list_id: [target words]} parsed from the memory export.
"""
from __future__ import annotations
import _bootstrap  # noqa: F401  (path shim, must be first)
import re
import json
from collections import defaultdict

import pandas as pd

import config
import common

WORDS_COL = "Words Shown: word (in_list, score, reaction_time)"


def parse_target_words(words_shown: str) -> list[str]:
    """Target = words flagged in_list = 't' in a cat_mem row's Words Shown cell."""
    if not isinstance(words_shown, str):
        return []
    out = []
    for w, flag in re.findall(r"([A-Za-z']+)\s*\(\s*([tf])", words_shown):
        if flag == "t":
            out.append(common.normalize_word(w))
    return out


def build_list_targets(mem: pd.DataFrame) -> dict[str, list[str]]:
    """Studied recall list per list NUMBER, keyed as 'list_<n>'.

    The b/c suffix on List Identifier is a recognition-probe variant, not the
    studied list: every presentation shows 12 words, and the studied (recall)
    list is the full 12-word set captured by the 'c' variant's in-list words.
    Scoring against this matches the hand scores (ICC ~0.97); the 5-6 word 'b'
    subsets do not. Verified against the research assistants' hand-scored recall file.
    """
    acc: dict[str, set] = defaultdict(set)
    for _, r in mem.iterrows():
        li = r.get("List Identifier")
        if not isinstance(li, str):
            continue
        num = re.match(r"list_(\d+)", li.strip())
        if num and li.strip().endswith("c"):
            for w in parse_target_words(r.get(WORDS_COL)):
                acc[f"list_{num.group(1)}"].add(w)
    return {k: sorted(v) for k, v in sorted(acc.items())}


def list_key(list_id) -> str | None:
    m = re.match(r"list_(\d+)", str(list_id).strip())
    return f"list_{m.group(1)}" if m else None


def normalize_cue(cue: str) -> str:
    # drop hyphens *and* any space straight after one ("Transpor- tation" -> "Transportation")
    return re.sub(r"\s+", " ", re.sub(r"-\s*", "", str(cue)).strip())


def build_index() -> pd.DataFrame:
    audio = common.index_audio_files()
    print(f"[index] unique audio basenames on disk: {len(audio)}")

    mem = common.read_zip_xlsx(config.MEM_SHEET)

    # ---- Word-list recall trials ----
    wl = build_list_targets(mem)
    rows = []
    for _, r in mem.iterrows():
        af = r.get("Recall Audio File Name")
        if not isinstance(af, str):
            continue
        target = wl.get(list_key(r.get("List Identifier"))) or parse_target_words(r.get(WORDS_COL))
        rows.append({
            "basename": af.replace(" ", "").strip(),
            "task": "word_recall",
            "stimulus": "|".join(sorted(set(target))),
            "list_id": r.get("List Identifier"),
            "watch_num_correct": r.get("Num Correct"),
            "watch_total_words": r.get("Total Num Words"),
        })
    mem_idx = pd.DataFrame(rows)

    idx = mem_idx.drop_duplicates("basename")

    # attach parsed metadata + resolved path
    meta = idx["basename"].apply(lambda b: common.parse_basename(b) or {})
    for col in ("participant_num", "device", "date", "time"):
        idx[col] = meta.apply(lambda d: d.get(col))
    idx = idx.merge(audio[["basename", "path"]], on="basename", how="left")
    idx["audio_found"] = idx["path"].notna()
    idx["datetime"] = pd.to_datetime(
        idx["date"].astype(str) + idx["time"].astype(str),
        format="%Y%m%d%H%M%S", errors="coerce",
    )

    # ---- write keys ----
    config.WORD_LISTS_JSON.write_text(json.dumps(wl, indent=2))
    print(f"[index] word lists: {len(wl)} -> {config.WORD_LISTS_JSON.name}")
    return idx


def main():
    idx = build_index()
    idx.to_csv(config.INDEX_CSV, index=False)
    found = int(idx["audio_found"].sum())
    print(f"[index] trials: {len(idx)} | audio resolved: {found} "
          f"({found/len(idx):.0%}) | unresolved: {len(idx)-found}")
    print(idx.groupby("task")["audio_found"].agg(["size", "sum"]))
    print(f"[index] wrote {config.INDEX_CSV}")


if __name__ == "__main__":
    main()
