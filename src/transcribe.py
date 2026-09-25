"""Transcribe task audio with local Whisper, caching one JSON per recording.

Cache key is the audio basename, so re-runs are free and the cache is portable.
Designed to run on CPU (slow) or GPU/Kamiak (fast). Use --limit to smoke-test.

    python src/transcribe.py --task word_recall --limit 20
    python src/transcribe.py --model small            # transcribe everything

Word onsets (needed by src/score_timing.py) require an explicit model AND a
separate cache, because every existing entry is a cache miss under
--word-timestamps and would otherwise be rewritten in place:

    SWC_TRANSCRIPT_CACHE=cache/transcripts_words \\
    python src/transcribe.py --model medium --word-timestamps

Schema 1 entries carry segment timings only; schema 2 adds per-word onsets
(`words`). A schema-1 entry is treated as a cache MISS only when onsets are
actually requested, so the upgrade pass is resumable and re-runs nothing it
does not have to.
"""
from __future__ import annotations
import _bootstrap  # noqa: F401  (path shim, must be first)
import json
import argparse
import inspect
from pathlib import Path

import pandas as pd

import config

_MODEL = None

TRANSCRIPT_SCHEMA = 2  # 1 = segment timings only; 2 = + per-word onsets

DEFAULT_CACHE = config.CACHE_DIR / "transcripts"

# Only used to refuse a silent quality downgrade of an existing cache entry.
_MODEL_RANK = {"tiny": 0, "base": 1, "small": 2, "medium": 3,
               "large": 4, "large-v1": 4, "large-v2": 4, "large-v3": 4}


def _get_model(name: str):
    global _MODEL
    if _MODEL is None:
        import whisper  # lazy: heavy import / optional dep
        print(f"[transcribe] loading whisper '{name}' ...")
        _MODEL = whisper.load_model(name)
    return _MODEL


def _supports_word_timestamps(model) -> bool:
    try:
        return "word_timestamps" in inspect.signature(model.transcribe).parameters
    except (TypeError, ValueError):
        return False


def cache_path(basename: str) -> Path:
    return config.TRANSCRIPT_CACHE / (basename + ".json")


def _segment_words(seg: dict) -> list[dict]:
    """Whisper attaches per-word timings to each segment as `words` when
    word_timestamps=True. Normalise to plain dicts; tolerate absence.
    Tokens that strip to empty are dropped, they would inflate n_words_timed
    and inject spurious near-zero inter-word intervals."""
    out = []
    for w in seg.get("words") or []:
        text = (w.get("word") or "").strip()
        if not text:
            continue
        out.append({
            "word": text,
            "start": w.get("start"),
            "end": w.get("end"),
            "probability": w.get("probability"),
        })
    return out


def has_word_schema(entry: dict) -> bool:
    """True if this entry came from a word-timestamp run, keyed off `schema`,
    NOT off truthiness of `words`. A silent recording legitimately yields
    schema 2 with words == [], and must not be re-transcribed forever."""
    try:
        return int(entry.get("schema", 1)) >= 2
    except (TypeError, ValueError):
        return False


def transcribe_one(path: str, basename: str, model_name: str, force=False,
                   word_timestamps: bool | None = None) -> dict:
    if word_timestamps is None:
        word_timestamps = config.WHISPER_WORD_TIMESTAMPS
    cp = cache_path(basename)
    if cp.exists() and not force:
        cached = json.loads(cp.read_text())
        # Only re-transcribe if we need word onsets and this entry predates them.
        if has_word_schema(cached) or not word_timestamps:
            return cached
        cached_model = cached.get("model")
        if (cached_model
                and _MODEL_RANK.get(cached_model, -1) > _MODEL_RANK.get(model_name, -1)):
            raise RuntimeError(
                f"refusing to overwrite {basename}: cached model "
                f"'{cached_model}' is better than requested '{model_name}'. "
                f"Re-run with --model {cached_model}, or --force to override.")

    model = _get_model(model_name)
    kw = dict(language=config.WHISPER_LANGUAGE, fp16=False)
    if word_timestamps:
        if not _supports_word_timestamps(model):
            raise RuntimeError(
                "installed whisper does not support word_timestamps; "
                "upgrade openai-whisper (pip install -U openai-whisper)")
        kw["word_timestamps"] = True
    # Deliberately NOT wrapped in try/except: a TypeError raised inside the DTW
    # word-alignment path is a real bug and must not be silently downgraded to a
    # schema-1 write that then gets counted as a successful upgrade.
    res = model.transcribe(path, **kw)

    segments, words = [], []
    for s in res.get("segments", []):
        sw = _segment_words(s)
        segments.append({"start": s["start"], "end": s["end"],
                         "text": s["text"], "words": sw})
        words.extend(sw)

    out = {
        "basename": basename,
        "text": res.get("text", "").strip(),
        "model": model_name,
        "schema": TRANSCRIPT_SCHEMA if word_timestamps else 1,
        "segments": segments,
        "words": words,
    }
    cp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def load_transcript(basename: str) -> str | None:
    cp = cache_path(basename)
    if cp.exists():
        return json.loads(cp.read_text()).get("text")
    return None


def load_words(basename: str) -> list[dict]:
    """Per-word onsets for one recording; [] if this entry predates schema 2."""
    cp = cache_path(basename)
    if not cp.exists():
        return []
    return json.loads(cp.read_text()).get("words", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="filter: word_recall|letter_fluency|category_fluency")
    ap.add_argument("--model", default=None,
                    help=f"whisper model (default {config.WHISPER_MODEL!r}); "
                         "must be explicit with --word-timestamps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None,
                    help="path to a text file of basenames (one per line); "
                         "transcribe only these")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--word-timestamps", action="store_true",
                    default=config.WHISPER_WORD_TIMESTAMPS,
                    help="ask Whisper for per-word onsets (required by score_timing.py)")
    ap.add_argument("--no-word-timestamps", dest="word_timestamps", action="store_false",
                    help="override SWC_WORD_TIMESTAMPS=1 from the environment")
    ap.add_argument("--allow-default-cache", action="store_true",
                    help="permit a word-timestamp run to write into the validated cache")
    args = ap.parse_args()

    # --- guard: never rewrite the validated cache with word-timestamp output ---
    if args.word_timestamps and not args.allow_default_cache:
        if config.TRANSCRIPT_CACHE.resolve() == DEFAULT_CACHE.resolve():
            ap.error(
                "refusing to write word-timestamped transcripts into the validated "
                f"cache:\n    {DEFAULT_CACHE}\n"
                "Every existing entry is a cache miss under --word-timestamps, so this "
                "would rewrite all of them in place.\nUse a fresh directory:\n"
                "    SWC_TRANSCRIPT_CACHE=cache/transcripts_words python src/transcribe.py "
                "--model medium --word-timestamps\n"
                "(or pass --allow-default-cache if that is genuinely what you want)")
    # --- guard: --model defaults to 'base'; the validated cache is 'medium' ---
    if args.word_timestamps and args.model is None:
        ap.error(
            f"--model must be explicit with --word-timestamps (it would default to "
            f"{config.WHISPER_MODEL!r}, and the validated cache is 'medium')")
    model_name = args.model or config.WHISPER_MODEL

    idx = pd.read_csv(config.INDEX_CSV)
    idx = idx[idx["audio_found"]]
    if args.task:
        idx = idx[idx["task"] == args.task]
    if args.only:
        want = {ln.strip() for ln in open(args.only) if ln.strip()}
        idx = idx[idx["basename"].isin(want)]
        missing = len(want) - len(idx)
        if missing:
            print(f"[transcribe] NOTE: {missing} of {len(want)} requested basenames "
                  f"are not in the index and will be skipped")
    if args.limit:
        idx = idx.head(args.limit)

    n = len(idx)
    print(f"[transcribe] {n} files | model={model_name} | "
          f"word_timestamps={args.word_timestamps} | cache={config.TRANSCRIPT_CACHE}")
    done = skipped = upgraded = failed = 0
    for i, (_, r) in enumerate(idx.iterrows(), 1):
        cp = cache_path(r["basename"])
        needs_upgrade = False
        if cp.exists() and not args.force:
            try:
                cached = json.loads(cp.read_text())
            except Exception:
                cached = {}
            if has_word_schema(cached) or not args.word_timestamps:
                skipped += 1
                continue
            needs_upgrade = True  # schema-1 entry, and we now want word onsets
        try:
            transcribe_one(r["path"], r["basename"], model_name,
                           force=args.force, word_timestamps=args.word_timestamps)
            done += 1
            upgraded += needs_upgrade
        except Exception as e:  # keep going; report at end
            failed += 1
            print(f"  ! {r['basename']}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{n} (new={done}, upgraded={upgraded}, "
                  f"cached={skipped}, failed={failed})")
    print(f"[transcribe] done. new={done}, upgraded={upgraded}, cached={skipped}, "
          f"failed={failed}, total cache="
          f"{len(list(config.TRANSCRIPT_CACHE.glob('*.json')))}")
    if failed:
        print(f"[transcribe] WARNING: {failed} file(s) failed, the pass is INCOMPLETE.")


if __name__ == "__main__":
    main()
