"""Score word-list recall from transcripts (deterministic, frozen recipe).

Per-token classification (Maureen's taxonomy, July 2026):
  correct              matched a studied-list word not yet credited
  repetition           repeat of an already-credited word or earlier intrusion
  intrusion-prior-list said word belongs to a list this participant studied earlier
  intrusion-new        said word matches no current or prior list
  deleted              not considered for scoring, but kept visible; the note
                       says why (filler, meta-speech, self-correction, retracted,
                       "no", recognizer artifact)

Frozen recipe rules (recipe version in recipe.RECIPE_VERSION), each recorded in
the token note so nothing is silently changed:
  R1  normalization, fillers and meta-speech patterns as in production
  R2  list-gated confusion map (ra_informed configuration only)
  R3  candidate rescue from the teacher-forced alternatives table
  R4  near-miss forcing in the first three content positions
  R5  compound split (exact halves, one-edit halves, studied + other-list half)
  R6  trailing low-confidence "you" is a recognizer artifact
  R7  retraction: a word followed by "no" loses credit; "no" is never an intrusion
  R8  self-correction statements are meta-speech; a repeated word they name is exempt
  R9  intrusions are counted only inside the recall span
  R10 a repeated intrusion is both a repetition and an intrusion (decision D2)
The legacy behaviour (baseline ICC 0.971) stays callable with legacy=True.

Version 2026-09-v2 (exploratory, SWC_RECIPE_VERSION=2026-09-v2 or the
recipe_version keyword; the frozen default is untouched) adds two rules,
each recorded in the token note:
  V1  loop guard: a verbatim copy of the preceding tokens that the word-level
      decode does not support (compared on a contraction-harmonised form) is
      a Whisper loop; its tokens are deleted before classification
      ("loop guard: dropped duplicate segment")
  V3  narration clauses: after every credit is known, intrusion tokens in a
      clause with no list word and no credited token, four or more content
      tokens and a pronoun or auxiliary are not counted
      ("narration clause: not counted")
A no-anchor intrusion fallback (V2) was tested and rejected on development
evidence; see recipe.py and docs/results/study2_v2_exploratory.md.

    python src/score_wordrecall.py                       # frozen scores -> outputs/
    python src/score_wordrecall.py --out-dir outputs/v2  # e.g. under the v2 switch
"""
from __future__ import annotations
import _bootstrap  # noqa: F401  (path shim, must be first)
import argparse
import os
import re
from pathlib import Path

import pandas as pd

import config
import common
import recipe

# ---------------------------------------------------------------------------
# Meta-speech patterns. Grounded in a scan of all 3,159 recall transcripts
# (432 matches; top phrases: "that's it", "i don't know", "that's all i can
# remember", "i don't remember any of the words from yesterday", ...).
# Applied per CLAUSE (split on , . ! ? ;) against the clause's token string,
# fillers still present. Reviewable/editable list, one pattern per line.
# ---------------------------------------------------------------------------
META_PATTERNS = [
    r"^(and |but |um |uh |oh )*that('?s| is| was)( about| pretty much)?( all| it| everything)\b.*",
    r"^(and |um |uh )*i (can'?t|cannot|don'?t|do not|couldn'?t|could not) (remember|recall|think)\b.*",
    r"^(and )*i (don'?t|do not) know\b.*",
    r"^(and )*i (think |guess |believe )?(that('?s| is| was) )?(all|it)( i (can )?(remember|recall|know|got|think of))?$",
    r"^what ('?s that|else|was|were|are|is)\b.*",
    r"^let me (think|see)\b.*",
    r"^(i'?m |i am )?drawing a blank\b.*",
    r"^i forgot( the rest| them| it| some)?$",
    r"^(there('?s| is| are) )?nothing else\b.*",
    r"^no more( words)?$",
    r"^(oh )?(gosh|shoot|boy|man|wow|geez|jeez)$",
    r"^(hmm+|umm+|uhh+|err+)$",
    r"^i (know|remember) (there ('?s|is|are|was|were) )?(more|others|some more)\b.*",
    r"^(these|those|the words?|they) (are|were|was|is)( the| all| from)?( same| new| different| yesterday'?s?)?\b.*words?.*",
    r"^from yesterday('?s)?( list)?$",
]
_META_RES = [re.compile(p) for p in META_PATTERNS]

INTRUSION_LABELS = ("intrusion-prior-list", "intrusion-new")
REPEATED_INTRUSION_NOTE = "repeated intrusion"


def is_repeated_intrusion(token: dict) -> bool:
    """R10: a repetition row that repeats an earlier intrusion (the note may
    carry the R9 "outside recall span" suffix)."""
    return token["classification"] == "repetition" and \
        token["note"].startswith(REPEATED_INTRUSION_NOTE)


def _clauses(transcript: str):
    """Split a transcript into clauses, each a list of normalized tokens
    (fillers kept, so meta patterns can see them)."""
    out = []
    for chunk in re.split(r"[.,!?;\n]+", str(transcript or "")):
        toks = common.tokens_from_text(chunk, drop_fillers=False)
        if toks:
            out.append(toks)
    return out


def _is_meta(clause_tokens) -> bool:
    s = " ".join(clause_tokens)
    return any(rx.match(s) for rx in _META_RES)


def _row(position, said, classification, matched="", note=""):
    return {"position": position, "word_said": said, "classification": classification,
            "matched_list_word": matched, "note": note}


def _match_in(tok, pool, lenient):
    for t in pool:
        if common.word_matches(tok, t, lenient):
            return t
    return None


# ---------------------------------------------------------------------------
# Legacy scorer (pre-freeze production; baseline ICC 0.971). Kept verbatim so
# regression tests and the validation report can reproduce the baseline.
# ---------------------------------------------------------------------------
def _classify_tokens_legacy(transcript, targets, lenient, prior_words):
    targets = [common.normalize_word(t) for t in targets if common.normalize_word(t)]
    prior = [common.normalize_word(w) for w in (prior_words or [])]
    prior = [w for w in prior if w and w not in targets]

    credited: set = set()
    seen: set = set()
    tokens = []
    pos = 0

    def emit_target_hit(p, said, hit, note=""):
        label = "repetition" if hit in credited else "correct"
        credited.add(hit)
        tokens.append(_row(p, said, label, hit, note))

    for clause in _clauses(transcript):
        meta = _is_meta(clause)
        for tok in clause:
            pos += 1
            if meta:
                tokens.append(_row(pos, tok, "deleted", note="meta-speech"))
                continue
            if tok in common._FILLERS:
                tokens.append(_row(pos, tok, "deleted", note="filler"))
                continue
            hit = _match_in(tok, targets, lenient)
            if hit is not None:
                emit_target_hit(pos, tok, hit)
                seen.add(tok)
                continue
            norm = common.normalize_word(tok)
            split = None
            for a in targets:
                if norm.startswith(a) and norm[len(a):] in targets:
                    split = (a, norm[len(a):])
                    break
            if split:
                emit_target_hit(pos, tok, split[0], note=f"compound split ({tok})")
                emit_target_hit(pos, tok, split[1], note=f"compound split ({tok})")
                seen.add(tok)
                continue
            if tok in seen:
                tokens.append(_row(pos, tok, "repetition", note="repeated intrusion"))
            else:
                phit = _match_in(tok, prior, lenient)
                if phit is not None:
                    tokens.append(_row(pos, tok, "intrusion-prior-list",
                                       note=f"prior-list word ({phit})"))
                else:
                    tokens.append(_row(pos, tok, "intrusion-new"))
            seen.add(tok)

    never = [t for t in targets if t not in credited]
    return tokens, never


# ---------------------------------------------------------------------------
# Frozen scorer
# ---------------------------------------------------------------------------
def classify_tokens(transcript: str, targets: list[str], lenient=True,
                    prior_words: list[str] | None = None, *,
                    basename: str | None = None,
                    recipe_config: str | None = None,
                    legacy: bool = False,
                    alternatives: list[dict] | None = None,
                    trailing_you: bool | None = None,
                    span_stop: str | None = None,
                    recipe_version: str | None = None,
                    word_decode: list[dict] | None = None):
    """Classify every spoken token under the frozen recipe (nothing silently dropped).

    Positional arguments are unchanged from the pre-freeze scorer. Keyword-only:
      basename      lets R3 read the alternatives table and R6 the word-level
                    cache for this recording; without it those two rules are
                    inert (no alternatives means no rescue, by design).
      recipe_config "automatic" or "ra_informed" (default config.RECIPE_CONFIG);
                    "automatic" disables the confusion map (R2) only.
      legacy        True runs the pre-freeze scorer for regression comparisons.
      alternatives  explicit alternatives rows (tests), overriding the table.
      trailing_you  explicit R6 verdict (tests), overriding the word-level cache.
      span_stop     "fillers" or "chatter": the R9 stop list (default
                    recipe.SPAN_STOP_DEFAULT, chosen by the margin rule).
      recipe_version "2026-09-freeze" (default recipe.RECIPE_VERSION, which
                    follows SWC_RECIPE_VERSION) or "2026-09-v2".
      word_decode   explicit word-level decode rows (tests) for the v2 loop
                    guard, overriding the word-level cache; ignored under
                    the frozen version.

    Returns (tokens, never_recalled):
      tokens: dicts {position, word_said, classification, matched_list_word, note}
              in spoken order; a compound split emits two rows sharing a position.
      never_recalled: studied words that were never credited.
    """
    if legacy:
        return _classify_tokens_legacy(transcript, targets, lenient, prior_words)
    cfg = recipe_config or config.RECIPE_CONFIG
    if cfg not in recipe.RECIPE_CONFIGS:
        raise ValueError(f"unknown recipe configuration {cfg!r}")
    span_stop = span_stop or recipe.SPAN_STOP_DEFAULT
    if span_stop not in recipe.SPAN_STOP_VARIANTS:
        raise ValueError(f"unknown span stop list {span_stop!r}")
    v2 = recipe.is_v2(recipe_version)

    targets = [common.normalize_word(t) for t in targets if common.normalize_word(t)]
    tset = set(targets)
    prior = [common.normalize_word(w) for w in (prior_words or [])]
    prior = [w for w in prior if w and w not in tset]

    clauses = _clauses(transcript)
    # V1 (v2 only): drop Whisper loops before anything else sees the stream.
    # The classification below runs on the shortened stream; original
    # positions are restored and the dropped tokens re-inserted as deleted
    # rows at the end, so nothing is silently lost.
    loop_dropped: list[int] = []
    orig_words: list[str] = []
    if v2:
        if word_decode is None:
            word_decode = recipe.load_word_decode(basename)
        orig_words = [tok for c in clauses for tok in c]
        clauses, loop_dropped = recipe.loop_guard_clauses(clauses, word_decode)
    meta_clause = [_is_meta(c) for c in clauses]
    flat = [(ci, i, tok) for ci, c in enumerate(clauses) for i, tok in enumerate(c)]
    words = [tok for _, _, tok in flat]
    n_flat = len(flat)

    # R8: self-correction statements over the same clause stream
    statements = recipe.find_statements(clauses)
    stmt_tokens: dict[int, dict] = {}
    for s in statements:
        for k in s["tok_idx"]:
            stmt_tokens[k] = s
    generic_starts = {s["tok_idx"][0]: s for s in statements if s["generic"]}

    # R7: which tokens are retracted by a following "no"
    def _retracted(k: int) -> bool:
        return (k + 1 < n_flat and words[k + 1] == "no"
                and not recipe.opens_giveup(words[k + 2:k + 6]))

    credited: set = set()
    seen: set = set()
    tokens: list[dict] = []

    def emit_target_hit(p, said, hit, note=""):
        label = "repetition" if hit in credited else "correct"
        credited.add(hit)
        tokens.append(_row(p, said, label, hit, note))

    def claim_preceding(before_idx: int):
        """R8 generic statement: the content token just before it (skipping
        deleted tokens and chatter, a few tokens back) is exempt from
        repetition if it currently is one."""
        k, steps = before_idx - 1, 0
        while k >= 0 and steps < recipe.SELFCORR_MAX_BACK:
            rows = [t for t in tokens if t["position"] == k + 1]
            if rows and rows[0]["classification"] != "deleted" \
                    and words[k] not in recipe.SELFCORR_CHATTER:
                if rows[0]["classification"] == "repetition":
                    for t in rows:
                        t["classification"] = "deleted"
                        t["note"] = "self-correction (named by a later statement)"
                return
            k -= 1
            steps += 1

    for k, (ci, i, tok) in enumerate(flat):
        pos = k + 1
        clause = clauses[ci]
        if k in generic_starts:
            claim_preceding(k)
        if meta_clause[ci]:
            tokens.append(_row(pos, tok, "deleted", note="meta-speech"))
            continue
        stmt = stmt_tokens.get(k)
        is_x = stmt is not None and stmt["x_idx"] == k
        if stmt is not None and not is_x:
            tokens.append(_row(pos, tok, "deleted", note="self-correction"))
            continue
        if tok == "no":
            giveup = recipe.opens_giveup(words[k + 1:k + 5])
            tokens.append(_row(pos, tok, "deleted",
                               note="no: opens a give-up phrase" if giveup
                               else "no: never an intrusion"))
            continue
        retracted = _retracted(k)

        hit = _match_in(tok, targets, lenient)
        note = ""
        if hit is None and cfg == "ra_informed":
            mapped = recipe.confusion_target(tok, clause, i, tset)
            if mapped is not None:
                hit, note = mapped, f"confusion map: {tok} -> {mapped}"
        if hit is None and tok in common._FILLERS:
            tokens.append(_row(pos, tok, "deleted", note="filler"))
            continue
        if hit is not None:
            if retracted:
                tokens.append(_row(pos, tok, "deleted", hit,
                                   note=(note + "; " if note else "") + "retracted"))
                continue
            if is_x and hit in credited:
                tokens.append(_row(pos, tok, "deleted", hit,
                                   note="self-correction (repeated word named)"))
                continue
            emit_target_hit(pos, tok, hit, note)
            seen.add(tok)
            continue

        # exact compound: token is exactly two studied words merged by ASR
        split = None
        for a in targets:
            if tok.startswith(a) and tok[len(a):] in tset:
                split = (a, tok[len(a):])
                break
        if split:
            if retracted:
                tokens.append(_row(pos, tok, "deleted", note="retracted"))
                continue
            emit_target_hit(pos, tok, split[0], note=f"compound split ({tok})")
            emit_target_hit(pos, tok, split[1], note=f"compound split ({tok})")
            seen.add(tok)
            continue

        # intrusions (prior-list vs new), repeats of intrusions (R10)
        if retracted:
            tokens.append(_row(pos, tok, "deleted", note="retracted"))
        elif tok in seen:
            if is_x:
                tokens.append(_row(pos, tok, "deleted",
                                   note="self-correction (repeated word named)"))
            else:
                tokens.append(_row(pos, tok, "repetition", note="repeated intrusion"))
        else:
            phit = _match_in(tok, prior, lenient)
            if phit is not None:
                tokens.append(_row(pos, tok, "intrusion-prior-list",
                                   note=f"prior-list word ({phit})"))
            else:
                tokens.append(_row(pos, tok, "intrusion-new"))
        seen.add(tok)

    # R6 trailing recognizer artifact
    if trailing_you is None:
        trailing_you = recipe.trailing_you_low(basename)
    if trailing_you and tokens and tokens[-1]["word_said"] == "you" \
            and tokens[-1]["classification"] != "deleted":
        tokens[-1].update(classification="deleted", matched_list_word="",
                          note="recognizer artifact")

    # R3 candidate rescue (alternatives table; absent table means no rescue)
    if alternatives is None:
        alternatives = recipe.load_alternatives(basename)
    _apply_rescue(tokens, alternatives, tset, credited, cfg)

    # R4 near-miss forcing in the first content positions
    content = [t for t in tokens if t["classification"] != "deleted"]
    for rank, t in enumerate(content[:recipe.NEAR_MISS_MAX_POSITION], 1):
        if t["classification"] not in INTRUSION_LABELS:
            continue
        w = t["word_said"]
        for tgt in sorted(tset - credited):
            if recipe.ed1(w, tgt):
                credited.add(tgt)
                t.update(classification="correct", matched_list_word=tgt,
                         note=f"near-miss (pos {rank}): {w} -> {tgt}")
                break

    # R5 compound split within one edit (studied + studied, then studied + other list)
    tokens = _apply_compounds(tokens, targets, tset, credited, prior)

    # R9 recall span (R10 counted inside it by score_recall)
    _mark_span(tokens, span_stop)

    if v2:
        _mark_narration(tokens, clauses, targets)
        tokens = _restore_loop_positions(tokens, loop_dropped, orig_words)

    never = [t for t in targets if t not in credited]
    return tokens, never


# ---------------------------------------------------------------------------
# Version 2026-09-v2 helpers (never called under the frozen version)
# ---------------------------------------------------------------------------
def _counted_intrusion(t: dict) -> bool:
    return t["classification"] in INTRUSION_LABELS or is_repeated_intrusion(t)


def _mark_narration(tokens, clauses, targets):
    """V3: intrusion tokens (and repeated intrusions) in a narration clause
    are flagged and not counted; nothing else about them changes.

    Runs after R3 to R5, so every credit is known: a clause holding any token
    that carries a matched list word (exact, lenient, confusion map, compound
    split, near miss or rescue) is never a narration clause, whatever the
    clause text alone would say."""
    clause_of, p = {}, 0
    for ci, c in enumerate(clauses):
        for _ in c:
            p += 1
            clause_of[p] = ci
    credited_clauses = {clause_of[t["position"]] for t in tokens
                        if t.get("matched_list_word") and t["position"] in clause_of}
    narr = {ci for ci, c in enumerate(clauses)
            if ci not in credited_clauses and recipe.is_narration_clause(c, targets)}
    for t in tokens:
        t["narration"] = False
        if narr and clause_of.get(t["position"]) in narr and _counted_intrusion(t):
            t["narration"] = True
            t["note"] = (t["note"] + "; " if t["note"] else "") + recipe.NARRATION_NOTE


def _restore_loop_positions(tokens, loop_dropped, orig_words):
    """Put the rows back on the original token positions and add one deleted
    row per token the loop guard removed."""
    if not loop_dropped:
        return tokens
    dead = set(loop_dropped)
    kept = [k for k in range(len(orig_words)) if k not in dead]
    for t in tokens:
        t["position"] = kept[t["position"] - 1] + 1
    for k in loop_dropped:
        tokens.append({**_row(k + 1, orig_words[k], "deleted", note=recipe.LOOP_NOTE),
                       "in_span": False, "narration": False})
    tokens.sort(key=lambda t: t["position"])
    return tokens


def _apply_rescue(tokens, alternatives, tset, credited, cfg):
    """R3: credit alt2/alt3 list words the model nearly wrote (one per list word).

    A rescue changes the first uncredited token whose written form matches the
    alternatives row; it never removes credit and never touches a token that
    already carries a list word.
    """
    rescued: set = set()
    for row in alternatives:
        cw = common.normalize_word(row.get("word", ""))
        if not cw or cw in tset:
            continue
        if cfg == "ra_informed" and recipe.HOMO.get(cw) in tset:
            continue  # the confusion map already handled this token
        if float(row.get("chosen_prob", 1.0)) >= recipe.RESCUE_CHOSEN_PROB_MAX:
            continue
        ent = row.get("entropy")
        if ent is not None and float(ent) > recipe.RESCUE_ENTROPY_MAX:
            continue
        for slot in recipe.RESCUE_ALT_SLOTS:
            w, p = recipe.parse_alternative(row.get(slot))
            if not w or p < recipe.RESCUE_ALT_PROB_MIN or w not in tset:
                continue
            if w in credited or w in rescued:
                continue
            target_row = next((t for t in tokens if t["word_said"] == cw
                               and not t["matched_list_word"]
                               and t["note"] != "recognizer artifact"), None)
            if target_row is None:
                continue
            credited.add(w)
            rescued.add(w)
            target_row.update(classification="correct", matched_list_word=w,
                              note=f"rescue: {row.get('word', cw).strip()} -> {w}")
            break


def _apply_compounds(tokens, targets, tset, credited, prior):
    """R5: an intrusion of COMPOUND_MIN_LEN+ letters within one edit of two
    concatenated studied words is credited as both (uncredited halves gain
    credit). Extension: studied word + a word from another CAT list credits the
    studied half and classifies the other half as an intrusion."""
    out = []
    others = sorted(recipe.all_list_words() - tset)
    for t in tokens:
        w = t["word_said"]
        if t["classification"] not in INTRUSION_LABELS or len(w) < recipe.COMPOUND_MIN_LEN:
            out.append(t)
            continue
        pair = None
        for a in sorted(tset):
            for b in sorted(tset):
                if a != b and recipe.within_one_edit(w, a + b) \
                        and (a not in credited or b not in credited):
                    pair = (a, b)
                    break
            if pair:
                break
        if pair:
            for half in pair:
                label = "repetition" if half in credited else "correct"
                credited.add(half)
                out.append(_row(t["position"], w, label, half,
                                note=f"compound: {w} -> {pair[0]}+{pair[1]}"))
            continue
        ext = None
        for s in sorted(tset - credited):
            for o in others:
                if recipe.within_one_edit(w, s + o) or recipe.within_one_edit(w, o + s):
                    ext = (s, o)
                    break
            if ext:
                break
        if ext:
            s, o = ext
            credited.add(s)
            out.append(_row(t["position"], w, "correct", s, note=f"compound: {w} -> {s}+{o}"))
            label = "intrusion-prior-list" if o in prior else "intrusion-new"
            out.append(_row(t["position"], w, label,
                            note=f"compound: {w} -> {s}+{o}; {o} is not on this list"))
            continue
        out.append(t)
    return out


def _mark_span(tokens, span_stop):
    """R9: mark intrusion tokens outside the recall span.

    The span runs from the first list-word token through the run of content
    tokens immediately after the last list-word token, stopping at the first
    deleted token (fillers, meta-speech) or, in the chatter variant, the first
    chatter word. Labels are kept; the note says the token is not counted.
    """
    listpos = [t["position"] for t in tokens if t["matched_list_word"]
               and t["classification"] in ("correct", "repetition")]
    if not listpos:
        start = end = None
    else:
        start, end = listpos[0], listpos[-1]
        for t in tokens:
            if t["position"] <= end:
                continue
            if t["classification"] == "deleted" or (
                    span_stop == "chatter" and t["word_said"] in recipe.SPAN_CHATTER):
                break
            end = t["position"]
    for t in tokens:
        inside = start is not None and start <= t["position"] <= end
        t["in_span"] = inside
        counted = t["classification"] in INTRUSION_LABELS or is_repeated_intrusion(t)
        if counted and not inside:
            t["note"] = (t["note"] + "; " if t["note"] else "") + "outside recall span"


def score_recall(transcript: str, targets: list[str], lenient=True,
                 prior_words: list[str] | None = None, **kw) -> dict:
    """Counts for one recording. Keyword arguments go to classify_tokens.

    n_intrusions follows the frozen convention (R9 span, R10 repeated
    intrusions counted); n_intrusions_all is the pre-freeze count of every
    intrusion token for comparison.
    """
    tokens, _never = classify_tokens(transcript, targets, lenient, prior_words, **kw)
    n = lambda label: sum(1 for t in tokens if t["classification"] == label)
    legacy = kw.get("legacy", False)
    if not legacy and recipe.is_v2(kw.get("recipe_version")):
        return _score_recall_v2(tokens, targets)
    n_correct = n("correct")
    n_prior_all, n_new_all = n("intrusion-prior-list"), n("intrusion-new")
    n_rep_intr = sum(1 for t in tokens if is_repeated_intrusion(t))
    if legacy:
        in_span = lambda t: True
    else:
        in_span = lambda t: t.get("in_span", False)
    n_prior = sum(1 for t in tokens if t["classification"] == "intrusion-prior-list" and in_span(t))
    n_new = sum(1 for t in tokens if t["classification"] == "intrusion-new" and in_span(t))
    n_rep_intr_span = sum(1 for t in tokens if is_repeated_intrusion(t) and in_span(t))
    n_target = len(targets)
    frozen_intrusions = n_prior + n_new + n_rep_intr_span
    return {
        "n_target": n_target,
        "n_correct": n_correct,
        "n_intrusions": (n_prior_all + n_new_all) if legacy else frozen_intrusions,
        "n_intrusions_prior": n_prior,
        "n_intrusions_new": n_new,
        "n_intrusions_all": n_prior_all + n_new_all,
        "n_intrusions_in_span": n_prior + n_new,
        "n_repeated_intrusions": n_rep_intr,
        "n_repeated_intrusions_in_span": n_rep_intr_span,
        "n_repetitions": n("repetition"),
        "n_deleted": n("deleted"),
        "n_said": sum(1 for t in tokens if t["classification"] != "deleted"),
        "pct_recalled": (n_correct / n_target) if n_target else float("nan"),
    }


def _score_recall_v2(tokens, targets) -> dict:
    """Counts under 2026-09-v2: the frozen columns with the V1 and V3 rules
    applied, plus the columns that show what the rules did. n_intrusions_all
    keeps its pre-freeze meaning (every intrusion token, narration included)."""
    n = lambda label: sum(1 for t in tokens if t["classification"] == label)
    narr = lambda t: t.get("narration", False)
    in_span = lambda t: t.get("in_span", False)
    n_correct = n("correct")
    n_prior_all, n_new_all = n("intrusion-prior-list"), n("intrusion-new")
    n_rep_intr = sum(1 for t in tokens if is_repeated_intrusion(t))
    n_prior = sum(1 for t in tokens if t["classification"] == "intrusion-prior-list"
                  and in_span(t) and not narr(t))
    n_new = sum(1 for t in tokens if t["classification"] == "intrusion-new"
                and in_span(t) and not narr(t))
    n_rep_intr_span = sum(1 for t in tokens if is_repeated_intrusion(t)
                          and in_span(t) and not narr(t))
    n_narration = sum(1 for t in tokens if narr(t))
    n_loop = sum(1 for t in tokens if t["note"] == recipe.LOOP_NOTE)
    n_intrusions = n_prior + n_new + n_rep_intr_span
    n_target = len(targets)
    return {
        "n_target": n_target,
        "n_correct": n_correct,
        "n_intrusions": n_intrusions,
        "n_intrusions_prior": n_prior,
        "n_intrusions_new": n_new,
        "n_intrusions_all": n_prior_all + n_new_all,
        "n_intrusions_in_span": n_prior + n_new,
        "n_repeated_intrusions": n_rep_intr,
        "n_repeated_intrusions_in_span": n_rep_intr_span,
        "n_repetitions": n("repetition"),
        "n_deleted": n("deleted"),
        "n_said": sum(1 for t in tokens if t["classification"] != "deleted"),
        "pct_recalled": (n_correct / n_target) if n_target else float("nan"),
        "n_loop_guard_dropped": n_loop,
        "n_intrusions_narration": n_narration,
    }


# ---------------------------------------------------------------------------
# Participant list history -> prior words per recording
# ---------------------------------------------------------------------------
def prior_words_map(idx: pd.DataFrame) -> dict:
    """basename -> list of words from lists this participant studied strictly
    before the recording's datetime (current list excluded via disjointness)."""
    import json
    from io_index import list_key
    wl = json.loads(config.WORD_LISTS_JSON.read_text())
    idx = idx.copy()
    idx["dt"] = pd.to_datetime(idx["datetime"], errors="coerce")
    out = {}
    for pid, g in idx.groupby("participant_num"):
        g = g.dropna(subset=["dt"]).sort_values("dt")
        first_seen = {}  # list key -> first administration time
        for _, r in g.iterrows():
            lk = list_key(r["list_id"])
            if lk and lk not in first_seen:
                first_seen[lk] = r["dt"]
        for _, r in g.iterrows():
            lk = list_key(r["list_id"])
            prior = [w for k, t0 in first_seen.items()
                     if k != lk and t0 < r["dt"] for w in wl.get(k, [])]
            out[r["basename"]] = prior
    return out


def resolve_out_dir(out_dir=None) -> Path:
    """Where the score tables go: --out-dir, else SWC_OUT_DIR, else outputs/.
    A run under a non-default recipe version must name its own directory so
    the frozen tables are never overwritten."""
    d = Path(out_dir or os.environ.get("SWC_OUT_DIR") or config.OUT_DIR)
    if recipe.RECIPE_VERSION != recipe.FROZEN_VERSION and d.resolve() == Path(config.OUT_DIR).resolve():
        raise SystemExit(f"refusing to write recipe {recipe.RECIPE_VERSION} scores over the "
                         f"frozen tables in {config.OUT_DIR}; pass --out-dir or set SWC_OUT_DIR")
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv=None):
    from transcribe import load_transcript
    ap = argparse.ArgumentParser(description="score word-list recall under the current recipe version")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default SWC_OUT_DIR or outputs/); required to differ "
                         "from outputs/ under a non-default SWC_RECIPE_VERSION")
    args = ap.parse_args(argv)
    out_dir = resolve_out_dir(args.out_dir)
    idx = pd.read_csv(config.INDEX_CSV)
    idx = idx[(idx["task"] == "word_recall") & idx["audio_found"]]
    priors = prior_words_map(idx)
    rows = []
    n_no_alternatives = 0
    for _, r in idx.iterrows():
        tr = load_transcript(r["basename"])
        if tr is None:
            continue
        targets = [t for t in str(r["stimulus"]).split("|") if t]
        if not recipe.load_alternatives(r["basename"]):
            n_no_alternatives += 1
        s = score_recall(tr, targets, prior_words=priors.get(r["basename"]),
                         basename=r["basename"])
        s.update({
            "basename": r["basename"],
            "participant_num": r["participant_num"],
            "datetime": r["datetime"],
            "list_id": r["list_id"],
            "watch_num_correct": r["watch_num_correct"],
            "recipe_version": recipe.RECIPE_VERSION,
            "recipe_config": config.RECIPE_CONFIG,
            "transcript": tr,
        })
        rows.append(s)
    out = pd.DataFrame(rows)
    dst = out_dir / "scores_word_recall.csv"
    out.to_csv(dst, index=False)
    print(f"[word_recall] scored {len(out)} trials -> {dst} "
          f"(recipe {recipe.RECIPE_VERSION}, config {config.RECIPE_CONFIG})")
    print(f"[word_recall] recordings without alternatives rows (no R3 rescue possible): "
          f"{n_no_alternatives} of {len(out)}")
    manifest = recipe.write_manifest("word_recall", {
        "n_scored": len(out),
        "n_without_alternatives": n_no_alternatives,
        "output": dst.name,
    }, manifest_path=out_dir / "recipe_manifest.json")
    print(f"[word_recall] manifest -> {manifest}")
    readme = recipe.write_exploratory_readme(out_dir)
    if readme:
        print(f"[word_recall] EXPLORATORY run under recipe {recipe.RECIPE_VERSION}; marker -> {readme}")
    if len(out):
        cols = ["n_correct", "n_intrusions", "n_intrusions_all",
                "n_repetitions", "n_deleted"]
        print(out[cols].describe().round(2))
    return out


if __name__ == "__main__":
    main()
