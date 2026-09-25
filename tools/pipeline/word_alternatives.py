#!/usr/bin/env python3
"""Diane's token-distribution reconstruction, adapted to our pipeline.

For each target recording: one teacher-forced forward pass per segment over the
tokens of the text our production transcription actually emitted (no fresh
decode, so the probabilities describe the transcripts we score). Emits, per
word: the model's probability for the word it chose, the top-5 alternative
tokens with probabilities, and the top-2 margin (low margin = the model was
torn between candidates).

    python3 tools/pipeline/word_alternatives.py --targets /tmp/alt_targets.txt
"""
from __future__ import annotations
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

import pandas as pd

import config

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default=None,
                    help="output CSV (default outputs/word_alternatives.csv)")
    args = ap.parse_args()

    import torch
    import whisper
    torch.set_num_threads(max(1, os.cpu_count() - 2))

    targets = [t.strip() for t in open(args.targets) if t.strip()]
    out = args.out or str(config.OUT_DIR / "word_alternatives.csv")
    # Resume support: a 4-hour run must survive being killed. Rows are appended
    # per recording, and already-written basenames are skipped on restart.
    import csv
    done = set()
    if os.path.exists(out):
        try:
            done = set(pd.read_csv(out, usecols=["basename"]).basename)
        except Exception:
            done = set()
    targets = [t for t in targets if t not in done]
    print(f"[alts] {len(done)} already done, {len(targets)} to go", flush=True)
    FIELDS = ["basename", "segment_start", "word", "chosen_prob", "entropy",
              "margin_top2", "alt1", "alt2", "alt3", "alt4", "alt5"]
    fh = open(out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS, restval="",
                            extrasaction="ignore")
    if not done and fh.tell() == 0:
        writer.writeheader()
    idx = pd.read_csv(config.INDEX_CSV)
    paths = dict(zip(idx["basename"], idx["path"]))
    wc = config.CACHE_DIR / "transcripts_words"

    print(f"[alts] loading whisper {args.model} (fp32, cpu) ...", flush=True)
    model = whisper.load_model(args.model)
    tok = whisper.tokenizer.get_tokenizer(
        multilingual=model.is_multilingual, language="en", task="transcribe")
    prefix = list(tok.sot_sequence_including_notimestamps)

    rows = []
    for bi, b in enumerate(targets, 1):
        cp = wc / (b + ".json")
        ap_ = paths.get(b)
        if not cp.exists() or not ap_ or not os.path.exists(ap_):
            continue
        entry = json.loads(cp.read_text())
        try:
            audio = whisper.load_audio(ap_)
        except Exception as e:
            print(f"  ! {b}: {e}", flush=True)
            continue
        for seg in entry.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            a = audio[int(seg["start"] * SR):]          # window starts at segment
            a = whisper.pad_or_trim(a)
            mel = whisper.log_mel_spectrogram(
                a, n_mels=model.dims.n_mels).to(model.device)
            gen = tok.encode(" " + text)
            tokens = torch.tensor([prefix + gen])
            with torch.no_grad():
                logits = model(mel.unsqueeze(0), tokens)
            probs = torch.softmax(logits.float(), dim=-1)[0]
            # probs[i] is the distribution for tokens[i+1]
            off = len(prefix) - 1
            # group generated tokens into words on leading-space boundaries
            words, cur = [], []
            for j, t in enumerate(gen):
                piece = tok.decode([t])
                if piece.startswith(" ") and cur:
                    words.append(cur)
                    cur = []
                cur.append(j)
            if cur:
                words.append(cur)
            for w in words:
                j0 = w[0]
                dist = probs[off + j0]
                chosen_p = float(dist[gen[j0]])
                # Diane's suggestion: distribution entropy, -sum p log p.
                # Low = peaked/confident, high = the model was guessing.
                entropy = float(-(dist * dist.clamp_min(1e-12).log()).sum())
                top_p, top_i = torch.topk(dist, k=args.topk + 1)
                alts, seen = [], set()
                for p, i_ in zip(top_p.tolist(), top_i.tolist()):
                    txt = tok.decode([i_]).strip()
                    if not txt or txt.lower() in seen:
                        continue
                    seen.add(txt.lower())
                    alts.append((txt, p))
                    if len(alts) == args.topk:
                        break
                margin = (alts[0][1] - alts[1][1]) if len(alts) > 1 else 1.0
                word_text = tok.decode([gen[k] for k in w]).strip()
                rows.append({
                    "basename": b,
                    "segment_start": round(float(seg["start"]), 2),
                    "word": word_text,
                    "chosen_prob": round(chosen_p, 4),
                    "entropy": round(entropy, 3),
                    "margin_top2": round(margin, 4),
                    **{f"alt{n+1}": f"{t} ({p:.2f})"
                       for n, (t, p) in enumerate(alts)},
                })
        for r in rows:
            writer.writerow(r)
        fh.flush()
        rows = []
        if bi % 5 == 0:
            print(f"  {bi}/{len(targets)} recordings", flush=True)

    fh.close()
    print(f"[alts] appended to {out}", flush=True)


if __name__ == "__main__":
    main()
