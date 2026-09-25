"""The frozen scoring recipe (version 2026-09-freeze): one source of truth.

Everything the scorers need that was fixed at the September 2026 freeze lives
here: the recipe version string, every threshold as a named constant with the
reason it has that value, the list-gated confusion map (R2), the edit-distance
helper behind near misses and compounds (R4, R5), the give-up and
self-correction grammars (R7, R8), the positional-span chatter list (R9), the
letter-fluency helpers (R11: digits, proper nouns) and the loaders for the two
side inputs the recall rules read (the teacher-forced alternatives table for R3
and the word-level transcript cache for R6). The scorers in score_wordrecall.py
and score_fluency.py apply these rules; the analysis scripts under
tools/analysis/ import the same objects so the prototypes and production cannot
drift apart.

Nothing here rewrites a transcript on disk and nothing writes under cache/.
"""
from __future__ import annotations
import _bootstrap  # noqa: F401  (path shim, must be first)
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

import common
import config

# Recipe versions. The frozen recipe is the default and the pre-specified
# result; "2026-09-v2" is the exploratory revision (V1 loop guard, V3
# narration clauses) that lives entirely behind this switch so the default
# keeps producing byte-identical output. Select with
# SWC_RECIPE_VERSION=2026-09-v2 or the recipe_version keyword of the scorers.
FROZEN_VERSION = "2026-09-freeze"
V2_VERSION = "2026-09-v2"
RECIPE_VERSIONS = (FROZEN_VERSION, V2_VERSION)
RECIPE_VERSION = os.environ.get("SWC_RECIPE_VERSION", FROZEN_VERSION)
if RECIPE_VERSION not in RECIPE_VERSIONS:
    raise ValueError(f"SWC_RECIPE_VERSION must be one of {RECIPE_VERSIONS}, got {RECIPE_VERSION!r}")
RECIPE_CONFIGS = ("automatic", "ra_informed")


def resolve_version(version: str | None = None) -> str:
    """The recipe version to score under: the argument, else the module default."""
    v = version or RECIPE_VERSION
    if v not in RECIPE_VERSIONS:
        raise ValueError(f"unknown recipe version {v!r}")
    return v


def is_v2(version: str | None = None) -> bool:
    return resolve_version(version) == V2_VERSION

# ---------------------------------------------------------------------------
# Thresholds (validated on the development set; see docs/results/validation_report.md)
# ---------------------------------------------------------------------------
# R3 candidate rescue: the written word must be one the model was unsure of
# (top-1 probability under one half) with a peaked enough distribution that the
# alternatives mean something (entropy at most 5 nats; above that the top-5 are
# noise). The candidate needs at least 5 percent probability; the ICC gain was
# flat across floors of 0.02 to 0.10, so the middle value was kept.
RESCUE_CHOSEN_PROB_MAX = 0.50
RESCUE_ENTROPY_MAX = 5.0
RESCUE_ALT_PROB_MIN = 0.05
RESCUE_ALT_SLOTS = ("alt2", "alt3")  # alt1 is the written word itself
# R4 near-miss forcing: only the first three spoken content positions, where
# P(correct) is 75 to 78 percent; from position four on it drops to 21 percent
# and the same rule at all positions lowered the ICC (0.974 versus 0.980).
NEAR_MISS_MAX_POSITION = 3
# R5 compound split: a merged token must be long enough that one edit from the
# concatenation of two four-letter list words cannot be a coincidence.
COMPOUND_MIN_LEN = 6
# R6 trailing recognizer artifact: Whisper's end-of-audio "you" carries a word
# probability far below any real word (real trailing words sit above 0.5).
TRAILING_YOU_PROB_MAX = 0.15
# R9 positional intrusions: two stop lists were tested. The fillers-only
# variant would be the default unless the chatter list beat it by more than
# CHATTER_ICC_MARGIN on the development set. It did (ICC 0.710 versus 0.685
# on the 601 RA intrusion scores, a gain of 0.025), so the chatter list is the
# frozen default; validate_recipe.py reports both every time it runs.
SPAN_CHATTER = frozenset({"oh", "yeah", "no", "sorry", "okay", "that", "thats",
                          "it", "you", "what", "not", "is", "was"})
SPAN_STOP_VARIANTS = ("fillers", "chatter")
SPAN_STOP_DEFAULT = "chatter"
CHATTER_ICC_MARGIN = 0.02
# R12 category fluency: the judge model the 0.910 token-level result was
# measured with; phrase level (0.836) stays behind the llm_phrase flag.
CATEGORY_JUDGE_MODEL = "gpt-5-mini"
CATEGORY_DEFAULT_MEMBERSHIP = "llm"
# Regression guard for tools/pipeline/validate_recipe.py --check.
CHECK_TOLERANCE = 0.002

THRESHOLDS = {
    "rescue_chosen_prob_max": RESCUE_CHOSEN_PROB_MAX,
    "rescue_entropy_max": RESCUE_ENTROPY_MAX,
    "rescue_alt_prob_min": RESCUE_ALT_PROB_MIN,
    "rescue_alt_slots": list(RESCUE_ALT_SLOTS),
    "near_miss_max_position": NEAR_MISS_MAX_POSITION,
    "compound_min_len": COMPOUND_MIN_LEN,
    "trailing_you_prob_max": TRAILING_YOU_PROB_MAX,
    "span_stop_default": SPAN_STOP_DEFAULT,
    "span_chatter": sorted(SPAN_CHATTER),
    "chatter_icc_margin": CHATTER_ICC_MARGIN,
    "category_judge_model": CATEGORY_JUDGE_MODEL,
    "category_default_membership": CATEGORY_DEFAULT_MEMBERSHIP,
}

# ---------------------------------------------------------------------------
# Version 2026-09-v2 only (exploratory; nothing below is read under the frozen
# default). The failure modes these rules address were noticed on the held-out
# Study 2 sample, so every threshold was chosen on the development corpus
# (recordings of the non-locked participants; the 602 RA-scored recordings
# for agreement) and none by Study 2 agreement. See
# docs/results/study2_v2_exploratory.md.
# ---------------------------------------------------------------------------
# V1 loop guard: a block of tokens that verbatim repeats the block just before
# it and that the word-timestamp decode of the same audio does not support is
# a Whisper decoding loop, not speech. Support floor: words inside the word
# decode's own looped segments have median probability 0.0 (83.5 percent
# below 0.05) against 1.4 percent of ordinary development words below 0.05;
# the touch count is flat from 0.01 to 0.2, so the floor is structural.
LOOP_SUPPORT_PROB_MIN = 0.05
# Minimum block length 1 (single-token stutters included): on the 602
# development recordings repetitions ICC 0.726 with 1 versus 0.690 (2) and
# 0.680 (3); the single-token touches are trailing stutters the word decode
# hears once and the RA did not count.
LOOP_MIN_BLOCK = 1
# Blocks anywhere in the transcript, not only trailing: development loops also
# sit mid-transcript (ICC 0.726 anywhere versus 0.671 trailing only).
LOOP_SCOPE = "anywhere"
LOOP_NOTE = "loop guard: dropped duplicate segment"
# The two decodes can tokenize the same speech differently (one writes "i'm",
# the other "I am"), which would leave a spoken contraction unsupported and a
# genuine repeat of it looking like a loop. Support is therefore compared on
# an expanded form: every token of both decodes is mapped through this table
# before the multisets are compared, so "i'm" and "i am" count as the same
# two words. Tokens outside the table compare as themselves. On the corpus at
# the time of writing the harmonised comparison drops exactly the same 1,053
# tokens as the plain one (none of the drops was contraction driven).
LOOP_CONTRACTIONS = {
    "i'm": "i am", "im": "i am", "i've": "i have", "ive": "i have", "i'd": "i would",
    "i'll": "i will", "you're": "you are", "you've": "you have", "you'll": "you will",
    "we're": "we are", "we've": "we have", "we'll": "we will", "they're": "they are",
    "they've": "they have", "that's": "that is", "thats": "that is", "it's": "it is",
    "there's": "there is", "here's": "here is", "what's": "what is", "let's": "let us",
    "he's": "he is", "she's": "she is", "who's": "who is", "where's": "where is",
    "don't": "do not", "dont": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "can not", "cant": "can not", "cannot": "can not", "couldn't": "could not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "isn't": "is not", "wasn't": "was not", "aren't": "are not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "ain't": "is not",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
}
# A no-anchor intrusion fallback (count intrusion tokens over the whole
# recording when no list word is credited, K=1) was built and rejected: on the
# 601 RA intrusion scores it alone moved ICC 0.711 to 0.322 (paired bootstrap
# difference -0.389, 95 percent CI [-0.562, -0.051]) and combined with V3 it
# was a tie (+0.010, CI [-0.099, +0.116]). It is not part of v2; see
# docs/results/study2_v2_exploratory.md.
# V3 narration clause: a clause inside the span with no list word, at least
# NARRATION_MIN_CONTENT content tokens and a first/second-person pronoun,
# auxiliary or cognition verb is participant narration, and its intrusion
# tokens are not counted (credits, near misses, rescue and the span itself are
# untouched). The rule runs after every credit is known and a clause holding a
# credited token of any kind (exact, lenient, confusion map, compound, near
# miss or rescue) is never narration, so a near-miss or rescue credit cannot
# sit inside a clause whose intrusions are discounted. Four content tokens
# rather than five: development ICC 0.718 versus 0.709 alone, and 22 of the 48
# development candidate clauses have exactly four; the RA never counted such
# a clause in full (0 of 6).
NARRATION_MIN_CONTENT = 4
NARRATION_NOTE = "narration clause: not counted"
# Fixed a priori (not tuned): first and second person pronouns and
# contractions, negated and plain auxiliaries and modals, cognition verbs of
# meta-speech. Covers 85.4 percent of development candidate clauses; the rest
# are bare noun runs (genuine intrusion lists) and Whisper noise, which stay
# counted by design.
NARRATION_PRON_AUX = frozenset({
    "i", "i'm", "i've", "i'd", "i'll", "im", "ive", "me", "my", "mine", "myself",
    "we", "we're", "we've", "our", "us", "you", "you're", "you've", "your",
    "can't", "cant", "cannot", "don't", "dont", "didn't", "didnt", "doesn't", "doesnt",
    "couldn't", "couldnt", "won't", "wont", "wouldn't", "wouldnt", "shouldn't", "isn't",
    "wasn't", "aren't", "weren't", "haven't", "hasn't", "hadn't", "ain't",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "gonna", "wanna", "gotta", "not",
    "think", "thought", "know", "knew", "remember", "remembered", "guess", "forgot",
    "forget", "recall", "believe", "mean", "said", "say", "sure", "trying", "try",
    "tried", "supposed",
})

THRESHOLDS_V2 = {
    "loop_support_prob_min": LOOP_SUPPORT_PROB_MIN,
    "loop_min_block": LOOP_MIN_BLOCK,
    "loop_scope": LOOP_SCOPE,
    "loop_contractions": dict(sorted(LOOP_CONTRACTIONS.items())),
    "narration_min_content": NARRATION_MIN_CONTENT,
    "narration_pron_aux": sorted(NARRATION_PRON_AUX),
}


def thresholds(version: str | None = None) -> dict:
    """The threshold block for the manifest: frozen thresholds, plus the v2
    block under "v2" when scoring under 2026-09-v2."""
    if is_v2(version):
        return {**THRESHOLDS, "v2": THRESHOLDS_V2}
    return dict(THRESHOLDS)


# ---------------------------------------------------------------------------
# R2 list-gated confusion map (RA-informed configuration only)
# ---------------------------------------------------------------------------
# Homophones and near-spellings Whisper plausibly emits for a studied word.
# A mapping fires ONLY when its value is on that recording's own 12-word list,
# so it cannot touch a recording that never studied the word. Pairs came from
# Maureen's error review (Aug 2026), Hallie's audio review (18 Aug) and
# Angela's (18 Aug). "still" -> "hill" is deliberately absent: "still" occurs
# inside real sentences ("I'm still thinking") and needs a clause-aware rule.
HOMO = {
    "jim": "gym", "gem": "gym", "gyn": "gym",
    "son": "sun", "sons": "sun",
    "bare": "bear",
    "rode": "road", "rowed": "road",
    "hare": "hair",
    "we'll": "wheel", "wheal": "wheel",
    "reign": "rain", "rein": "rain",
    "cache": "cash",
    "bred": "bread",
    "teem": "team",
    "bawl": "ball",
    "shoo": "shoe",
    "quay": "key",
    "cede": "seed",
    "wring": "ring",
    "sox": "sock",
    "bow": "bowl", "fentz": "fence", "hen": "pen",
    "shu": "shoe", "teen": "team",
    "dutch": "duck", "dow": "doll", "red": "bread",
}

_CLAUSE_SPLIT = re.compile(r"([.,!?;\n]+)")


def confusion_target(tok: str, clause: list[str], i: int, targets) -> str | None:
    """List word that `tok` (normalized, at index i of its clause) maps to, or None.

    Token-level form of the map: a clause that is exactly "I" is a spoken
    "eye" (a pronoun attaches to a verb); "thank" is "bank" unless "you"
    follows (a hallucinated "thank you" must never be converted); any other
    key fires only when its value is on the list.
    """
    if tok == "i":
        return "eye" if ("eye" in targets and clause == ["i"]) else None
    if tok == "thank":
        nxt = clause[i + 1] if i + 1 < len(clause) else ""
        return "bank" if ("bank" in targets and nxt != "you") else None
    tgt = HOMO.get(tok)
    return tgt if tgt is not None and tgt in targets else None


def lone_i_to_eye(text: str) -> str:
    """Text-level variant kept for the analysis scripts: a clause that is
    exactly the token "i" becomes "eye"."""
    parts = _CLAUSE_SPLIT.split(str(text))
    out = []
    for p in parts:
        toks = common.tokens_from_text(p, drop_fillers=False)
        out.append(" eye" if toks == ["i"] else p)
    return "".join(out)


def apply_homo(text: str, targets) -> str:
    """Text-level variant of R2 for the analysis scripts (the scorer works on
    tokens through confusion_target and never rewrites the transcript)."""
    t = str(text)
    if "eye" in targets:
        t = lone_i_to_eye(t)
    for h, tgt in HOMO.items():
        if tgt in targets:
            t = re.sub(rf"(?i)(?<![\w']){re.escape(h)}(?![\w'])", tgt, t)
    if "bank" in targets:
        t = re.sub(r"(?i)(?<![\w'])thank(?!\s+you)(?![\w'])", "bank", t)
    return t


# ---------------------------------------------------------------------------
# R4, R5 edit distance
# ---------------------------------------------------------------------------
def ed1(a: str, b: str) -> bool:
    """True if a and b differ by exactly one substitution, insertion or deletion."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            diff += 1
            if diff > 1:
                return False
            j += 1
    return True


def within_one_edit(a: str, b: str) -> bool:
    """True if a equals b or is exactly one edit away (R5 reads "within one
    edit" as zero or one, so an exact merge such as keystone also splits)."""
    return a == b or ed1(a, b)


@lru_cache(maxsize=1)
def all_list_words() -> frozenset:
    """Every word on any CAT study list (normalized), for the R5 extension."""
    wl = json.loads(config.WORD_LISTS_JSON.read_text())
    return frozenset(common.normalize_word(w) for ws in wl.values() for w in ws)


# ---------------------------------------------------------------------------
# R7 retraction: give-up phrases
# ---------------------------------------------------------------------------
# A "no" that opens "no, I can't remember" is giving up, not retracting the
# previous word. Text form (used by the analysis script) and token form.
GIVEUP = re.compile(
    r"^[,.!?;\s]*(i\s+)?(can'?t|cannot|don'?t|do not)\s+(remember|recall|think)",
    re.IGNORECASE)
_GIVEUP_TOKENS = re.compile(r"^(i )?(can'?t|cannot|don'?t|do not) (remember|recall|think)")


def opens_giveup(following: list[str]) -> bool:
    """True if the normalized tokens after a "no" start a give-up phrase."""
    return bool(_GIVEUP_TOKENS.match(" ".join(following[:4])))


# ---------------------------------------------------------------------------
# R8 self-correction statements (scoring binder, RAVLT section)
# ---------------------------------------------------------------------------
GENERIC = {"that", "this", "it", "them", "those", "these", "one", "some",
           "something", "myself", "a", "the", "any", "words", "couple"}
# Words skipped when a generic statement looks back for the word it refers to.
SELFCORR_CHATTER = {"oh", "yeah", "yes", "no", "sorry", "okay", "ok", "um", "uh",
                    "hmm", "well", "right", "wait"}
SELFCORR_MAX_BACK = 4
_X = r"(?P<x>[a-z']+)"
_TAIL = (r"(?: (?:already|before|once|twice|one|word|again|earlier|previously"
         r"|yet|or|not|too|i|think|guess|a|couple|of|times|few|words|that"
         r"|maybe|probably|already))*")
_PRE = r"(?:(?:oh|um|uh|yeah|yes|no|okay|ok|and|but|so|well|hmm|wait) )*"
_SUBJ = r"i(?:'ve| have|'d| had)?"
_HEDGE = r"(?:(?:i )?(?:think|guess|know|hope|believe|bet|thought|feel like) (?:that )?)?"
_ADV = r"(?:(?:already|just|actually|even) )?"
_VERB = r"(?:said|mentioned|repeated)"
STATEMENT_RES = [
    # did I (already) say X (already)?  / have I said X?
    re.compile(rf"(?:did|have|had|didn't|haven't) i {_ADV}(?:say|said|mention|repeat) {_X}{_TAIL}(?= |$)"),
    # I (think I) (already) said X (already|before) / I've said X
    re.compile(rf"{_HEDGE}{_SUBJ} {_ADV}{_VERB} {_X}{_TAIL}(?= |$)"),
    # I may/might have (already) said X
    re.compile(rf"{_HEDGE}{_SUBJ} (?:may|might|must|could|probably) (?:have |already )*{_VERB} {_X}{_TAIL}(?= |$)"),
    # I don't know if I said X (or not) / not sure whether I said X
    re.compile(rf"(?:i'm |i am |i )?(?:don't|dont|do not|not sure|wasn't sure|unsure) (?:know )?(?:if|whether) {_SUBJ} {_ADV}{_VERB} {_X}{_TAIL}(?= |$)"),
    # clause-initial: already said X / said X
    re.compile(rf"^{_PRE}(?:(?:already|just) )?{_VERB} {_X}{_TAIL}$"),
]
# "I think I did", answering a self-correction question in the previous clause
ANSWER_RE = re.compile(rf"^{_PRE}(?:i )?(?:think |guess |hope |believe )?(?:i )?(?:did|didn't|did not|do|don't|doubt it)(?: (?:i think|already|so))?$")


def find_statements(clauses: list[list[str]]) -> list[dict]:
    """Locate self-correction statements in a clause list (normalized tokens,
    fillers kept). Token indices are flat positions over all clauses, so the
    scorer can line them up with its own token stream.

    Returns dicts {tok_idx: [flat indices], x_idx: flat index or None,
    generic: bool, x_word: str, text: str}. A generic statement ("I said
    that") names no word; the scorer then claims the preceding content token.
    """
    stmts = []
    offset = 0
    prev_clause_had = False
    for words in clauses:
        joined = " ".join(words)
        offs, c = [], 0
        for w in words:
            offs.append(c)
            c += len(w) + 1
        found = []
        for rx in STATEMENT_RES:
            for m in rx.finditer(joined):
                if m.start() > 0 and joined[m.start() - 1] != " ":
                    continue
                i0 = offs.index(m.start())
                i1 = max(k for k, o in enumerate(offs) if o < m.end())
                xi = offs.index(m.start("x"))
                found.append((i0, i1, xi))
        found.sort(key=lambda f: (f[0], -(f[1] - f[0])))
        kept, last_end = [], -1
        for f in found:
            if f[0] <= last_end:
                continue
            kept.append(f)
            last_end = f[1]
        if not kept and prev_clause_had and ANSWER_RE.match(joined):
            kept = [(0, len(words) - 1, None)]
        for i0, i1, xi in kept:
            x_word = words[xi] if xi is not None else ""
            generic = xi is None or x_word in GENERIC or x_word in common._FILLERS
            stmts.append({"tok_idx": [offset + k for k in range(i0, i1 + 1)],
                          "x_idx": None if generic else offset + xi,
                          "generic": generic, "x_word": x_word,
                          "text": " ".join(words[i0:i1 + 1])})
        prev_clause_had = bool(kept)
        offset += len(words)
    return stmts


# ---------------------------------------------------------------------------
# R11 letter fluency helpers: digits and proper nouns
# ---------------------------------------------------------------------------
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_DIGITS_RE = re.compile(r"(?<![\w-])(\d{1,4})(?![\w-])")


def number_to_words(n: int) -> str | None:
    """English words for 0 to 9999 ("10" -> "ten", "21" -> "twenty one",
    "1000" -> "one thousand"); None outside that range."""
    if n < 0 or n > 9999:
        return None
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (f" {_ONES[o]}" if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{_ONES[h]} hundred" + (f" {number_to_words(r)}" if r else "")
    k, r = divmod(n, 1000)
    return f"{_ONES[k]} thousand" + (f" {number_to_words(r)}" if r else "")


def digits_to_words(text: str) -> tuple[str, int]:
    """Replace standalone digit strings with number words so the tokenizer
    (which strips digits) can score them. Returns (text, n_converted)."""
    count = 0

    def sub(m):
        nonlocal count
        words = number_to_words(int(m.group(1)))
        if words is None:
            return m.group(0)
        count += 1
        return words
    return _DIGITS_RE.sub(sub, str(text or "")), count


@lru_cache(maxsize=1)
def load_proper_nouns(path: Path | None = None) -> dict[str, str]:
    """word -> effective tier from keys/proper_nouns.csv.

    Tier "exclude" is a proper noun with no ordinary reading (never credited,
    decision D1); "review" has an ordinary reading and is credited but flagged.
    An RA decision in the ra_decision column ("exclude" or "keep") overrides the
    judge's tier once the adjudication comes back; "keep" drops the word from
    the map. Missing file: empty map (nothing excluded, nothing flagged).
    """
    p = Path(path) if path else config.PROPER_NOUNS_CSV
    out: dict[str, str] = {}
    if not p.exists():
        return out
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            word = common.normalize_word(row.get("word", ""))
            tier = (row.get("tier") or "").strip().lower()
            decision = (row.get("ra_decision") or "").strip().lower()
            if decision in ("exclude", "keep"):
                tier = decision
            if word and tier in ("exclude", "review"):
                out[word] = tier
    return out


# ---------------------------------------------------------------------------
# Side inputs for R3 and R6
# ---------------------------------------------------------------------------
_ALT_PAT = re.compile(r"^(.*)\s+\(([\d.]+)\)$")


def parse_alternative(cell) -> tuple[str | None, float]:
    """"Pin (0.12)" -> ("pin", 0.12); (None, 0.0) for blanks."""
    m = _ALT_PAT.match(str(cell))
    if not m:
        return None, 0.0
    return common.normalize_word(m.group(1)), float(m.group(2))


@lru_cache(maxsize=1)
def _alternatives_table() -> dict[str, list[dict]]:
    """basename -> alternatives rows (teacher-forced pass), read once. The two
    CSVs are concatenated and de-duplicated on (basename, segment_start, word)."""
    out: dict[str, list[dict]] = {}
    seen: set = set()
    for p in config.ALTERNATIVES_CSVS:
        if not Path(p).exists():
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("basename"), row.get("segment_start"), row.get("word"))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    rec = {"word": row.get("word", ""),
                           "chosen_prob": float(row.get("chosen_prob") or 0.0),
                           "entropy": float(row["entropy"]) if row.get("entropy") else None}
                except ValueError:
                    continue
                for slot in RESCUE_ALT_SLOTS:
                    rec[slot] = row.get(slot, "")
                out.setdefault(row.get("basename", ""), []).append(rec)
    return out


def load_alternatives(basename: str | None) -> list[dict]:
    """Alternatives rows for one recording ([] when the pass never covered it)."""
    if not basename:
        return []
    return _alternatives_table().get(basename, [])


def alternatives_coverage() -> int:
    """Number of recordings the alternatives tables cover."""
    return len(_alternatives_table())


def trailing_you_low(basename: str | None) -> bool:
    """True if the last decoded word in the word-level cache is "you" with
    probability under TRAILING_YOU_PROB_MAX (Whisper's end-of-audio artifact).
    False when the word-level file is missing or unreadable."""
    if not basename:
        return False
    p = config.WORDS_CACHE / (basename + ".json")
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        words = [w for s in d.get("segments", []) for w in s.get("words", [])]
        return bool(words) and common.normalize_word(words[-1]["word"]) == "you" \
            and float(words[-1].get("probability", 1.0)) < TRAILING_YOU_PROB_MAX
    except (OSError, ValueError, KeyError, TypeError):
        return False


# ---------------------------------------------------------------------------
# V1 loop guard (2026-09-v2 only): Whisper decoding loops
# ---------------------------------------------------------------------------
def load_word_decode(basename: str | None) -> list[dict]:
    """Every word of the word-timestamp decode for one recording, in order
    ({word, start, end, probability}); [] when the file is missing or
    unreadable, which makes the loop guard inert for that recording."""
    if not basename:
        return []
    p = config.WORDS_CACHE / (basename + ".json")
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
        words = [w for s in d.get("segments", []) for w in s.get("words", [])]
        return words or list(d.get("words") or [])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []


def loop_compare_form(token: str) -> tuple[str, ...]:
    """The form a token takes in the loop guard's support comparison: its
    expansion under LOOP_CONTRACTIONS, or the token itself."""
    return tuple(LOOP_CONTRACTIONS.get(token, token).split())


def compare_counter(tokens) -> Counter:
    """Multiset of compare forms (expanded words) over a token stream."""
    c: Counter = Counter()
    for t in tokens:
        for w in loop_compare_form(t):
            c[w] += 1
    return c


def supported_tokens(words: list[dict], p_min: float = LOOP_SUPPORT_PROB_MIN) -> Counter:
    """Multiset of compare forms the word decode supports: every word with
    probability at least p_min, tokenized as the production transcript is,
    mapped through loop_compare_form and counted over the whole recording
    (position blind; the word decode stacks its own looped copies at the
    window edge with probability near zero, so the floor already separates
    them)."""
    toks: list[str] = []
    for x in words:
        try:
            prob = float(x.get("probability", 1.0))
        except (TypeError, ValueError):
            prob = 1.0
        if prob >= p_min:
            toks.extend(common.tokens_from_text(x.get("word", ""), drop_fillers=False))
    return compare_counter(toks)


def find_loop_blocks(tokens: list[str], excess: Counter, min_block: int = LOOP_MIN_BLOCK,
                     scope: str = LOOP_SCOPE) -> list[tuple[int, int]]:
    """Blocks of the token list that the loop guard deletes.

    Walking backwards from the end, a block of L >= min_block consecutive
    tokens that equals the L tokens immediately before it is deleted when
    every one of its tokens is still in `excess` (the multiset of compare
    forms, see loop_compare_form, that the production transcript has beyond
    what the word decode supports). The
    largest block ending at the latest position goes first; the search
    repeats on the shortened list until nothing qualifies. Returns the
    deleted blocks as (start, length) pairs on the ORIGINAL token positions;
    the first occurrence of anything is never deleted.
    """
    alive = list(range(len(tokens)))   # original index of each surviving token
    toks = list(tokens)
    rem = Counter(excess)
    blocks: list[tuple[int, int]] = []
    while True:
        found = None
        ends = [len(toks)] if scope == "trailing" else range(len(toks), 0, -1)
        for e in ends:
            for L in range(e // 2, min_block - 1, -1):
                blk = toks[e - L:e]
                if blk != toks[e - 2 * L:e - L]:
                    continue
                need = compare_counter(blk)
                if all(rem[t] >= n for t, n in need.items()):
                    found = (e - L, L)
                    break
            if found:
                break
        if not found:
            return blocks
        s, L = found
        rem -= compare_counter(toks[s:s + L])
        blocks.append((alive[s], L))
        del toks[s:s + L]
        del alive[s:s + L]


def loop_guard_clauses(clauses: list[list[str]], words: list[dict], *,
                       p_min: float = LOOP_SUPPORT_PROB_MIN,
                       min_block: int = LOOP_MIN_BLOCK,
                       scope: str = LOOP_SCOPE) -> tuple[list[list[str]], list[int]]:
    """Apply the loop guard to a clause stream (lists of normalized tokens,
    fillers kept, as score_wordrecall._clauses builds it).

    Returns (kept_clauses, dropped): the clauses with the looped tokens
    removed (clauses left empty are dropped) and the flat indices of the
    removed tokens over the original stream. Inert (nothing removed) when
    the word decode is missing or empty.
    """
    if not words:
        return [list(c) for c in clauses], []
    flat = [t for c in clauses for t in c]
    excess = compare_counter(flat) - supported_tokens(words, p_min)
    if not excess:
        return [list(c) for c in clauses], []
    dead: set[int] = set()
    for start, L in find_loop_blocks(flat, excess, min_block, scope):
        dead.update(range(start, start + L))
    if not dead:
        return [list(c) for c in clauses], []
    kept, k = [], 0
    for c in clauses:
        keep = [t for j, t in enumerate(c) if (k + j) not in dead]
        k += len(c)
        if keep:
            kept.append(keep)
    return kept, sorted(dead)


_CHUNK_RE = re.compile(r"[^.,!?;\n\s]+")
_CLAUSE_PUNCT_RE = re.compile(r"[.!?;\n]")


def loop_guard_text(text: str, words: list[dict], **kw) -> tuple[str, list[str]]:
    """Text-level form of the loop guard for the analysis scripts: the same
    rule applied to the raw transcript, returning (new_text, dropped_tokens).
    Chunks are the maximal runs the scorer tokenizes, so the kept text
    tokenizes to exactly the kept tokens; clause punctuation on a dropped
    chunk moves to the previous surviving chunk when that one has none."""
    text = str(text or "")
    chunks = [(m.start(), m.end(), common.normalize_word(m.group(0)))
              for m in _CHUNK_RE.finditer(text)]
    tok_chunks = [i for i, c in enumerate(chunks) if c[2]]
    clauses = [[chunks[i][2] for i in tok_chunks]]
    _kept, dropped = loop_guard_clauses(clauses, words, **kw)
    if not dropped:
        return text, []
    dead = {tok_chunks[j] for j in dropped}
    out: list[str] = []
    pos, last_delim = 0, None
    for i, (a, b, _tok) in enumerate(chunks):
        nxt = chunks[i + 1][0] if i + 1 < len(chunks) else len(text)
        delim = text[b:nxt]
        if i in dead:
            if last_delim is not None and _CLAUSE_PUNCT_RE.search(delim) \
                    and not _CLAUSE_PUNCT_RE.search(out[last_delim]):
                out[last_delim] = delim
            pos = nxt
            continue
        out.append(text[pos:a] if pos < a else "")
        out.append(text[a:b])
        out.append(delim)
        last_delim = len(out) - 1
        pos = nxt
    new = re.sub(r"\s+", " ", "".join(out)).strip()
    return new, [clauses[0][j] for j in dropped]


# ---------------------------------------------------------------------------
# V3 narration clause (2026-09-v2 only)
# ---------------------------------------------------------------------------
def clause_has_list_word(clause: list[str], targets: list[str]) -> bool:
    """True if any token of the clause is a list word by lenient match, by the
    list-gated confusion map or as an exact compound of two list words."""
    tset = set(targets)
    for i, tok in enumerate(clause):
        if any(common.word_matches(tok, t, True) for t in targets):
            return True
        if confusion_target(tok, clause, i, tset) is not None:
            return True
        for a in targets:
            if tok.startswith(a) and tok[len(a):] in tset:
                return True
    return False


def is_narration_clause(clause: list[str], targets: list[str],
                        min_content: int = NARRATION_MIN_CONTENT) -> bool:
    """V3: at least min_content non-filler tokens, at least one pronoun,
    auxiliary or cognition verb, and no list word."""
    if sum(1 for t in clause if t not in common._FILLERS) < min_content:
        return False
    if not any(t in NARRATION_PRON_AUX for t in clause):
        return False
    return not clause_has_list_word(clause, targets)


# ---------------------------------------------------------------------------
# Exploratory output marker (any version other than the frozen one)
# ---------------------------------------------------------------------------
README_NAME = "README.md"


def write_exploratory_readme(out_dir: Path, version: str | None = None,
                             extra: str = "") -> Path | None:
    """Write README.md into an output directory produced under a non-frozen
    recipe version, stating EXPLORATORY and the version, so the directory can
    never be mistaken for the frozen result. Returns None (and writes nothing)
    under the frozen version."""
    v = resolve_version(version)
    if v == FROZEN_VERSION:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    text = (
        f"# EXPLORATORY: recipe version {v}\n\n"
        f"Every file in this directory was produced under recipe version `{v}`, "
        f"not the frozen recipe `{FROZEN_VERSION}`. These numbers are exploratory "
        "(the rules of this version were motivated by inspection of the held-out "
        "Study 2 sample) and are not the pre-specified result. The frozen result "
        "lives in `outputs/` and `outputs/study2/` (copied to `docs/results/study2/`); "
        "report those.\n\n"
        f"Written {stamp} by the scoring pipeline.\n"
    )
    if extra:
        text += "\n" + extra.rstrip() + "\n"
    dst = out_dir / README_NAME
    dst.write_text(text)
    return dst


# ---------------------------------------------------------------------------
# R15 manifest
# ---------------------------------------------------------------------------
MANIFEST_PATH = config.OUT_DIR / "recipe_manifest.json"


def git_commit() -> str | None:
    """Current git commit of the repo, or None outside a checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT,
                             capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha1_dir(d: Path, suffix: str = ".json") -> dict:
    """Aggregate hash of a cache directory: sha1 over (name, file sha1) pairs."""
    files = sorted(p for p in d.glob(f"*{suffix}"))
    h = hashlib.sha1()
    for p in files:
        h.update(p.name.encode())
        h.update(_sha1_file(p).encode())
    return {"n_files": len(files), "sha1": h.hexdigest()}


def input_hashes() -> dict:
    """Content hashes of every input the frozen recipe reads (no data inside)."""
    out: dict = {}
    for p in [config.INDEX_CSV, config.WORD_LISTS_JSON, config.FLUENCY_CUES_JSON,
              config.PROPER_NOUNS_CSV, config.HAND_MEM_CSV, config.HAND_LFCF_CSV,
              *config.ALTERNATIVES_CSVS]:
        p = Path(p)
        out[p.name] = _sha1_file(p) if p.exists() else None
    for d in (config.TRANSCRIPT_CACHE, config.WORDS_CACHE):
        d = Path(d)
        out[d.name] = _sha1_dir(d) if d.exists() else None
    return out


def write_manifest(step: str, info: dict, recipe_config: str | None = None,
                   manifest_path: Path | None = None) -> Path:
    """Record (or update) outputs/recipe_manifest.json for one pipeline step.

    The manifest carries the recipe version and configuration, every
    threshold, the git commit and the input hashes; each scoring step adds its
    own block under "steps" so a later reader can see what was run and when.
    A scorer writing to another output directory (a 2026-09-v2 run) passes
    its own manifest_path so the frozen manifest is never touched.
    """
    cfg = recipe_config or config.RECIPE_CONFIG
    path = Path(manifest_path) if manifest_path else MANIFEST_PATH
    manifest = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text())
        except ValueError:
            manifest = {}
    if manifest.get("recipe_version") != RECIPE_VERSION or manifest.get("recipe_config") != cfg:
        manifest = {}
    manifest.update({
        "recipe_version": RECIPE_VERSION,
        "recipe_config": cfg,
        "thresholds": thresholds(),
        "git_commit": git_commit(),
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": input_hashes(),
    })
    steps = manifest.setdefault("steps", {})
    steps[step] = {"generated_at": manifest["generated_at"], **info}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_manifest() -> dict | None:
    """The current manifest, or None if none has been written."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except ValueError:
        return None
