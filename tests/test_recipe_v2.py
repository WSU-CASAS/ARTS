"""Synthetic-transcript tests for recipe version 2026-09-v2 (exploratory).

The v2 rules (V1 loop guard, V3 narration clauses) are exercised on made-up
transcripts against a made-up word list, and the same inputs are scored under
the frozen version to show it is unchanged. The word-level decode the loop
guard reads is injected through the word_decode keyword; no recording,
participant or hand score is used. Every call names its recipe version, so
the suite gives the same result whatever SWC_RECIPE_VERSION is set to.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

import recipe
from score_wordrecall import classify_tokens, score_recall

LIST = ["gym", "eye", "bank", "dime", "wind", "stone", "pen", "log", "duck",
        "doll", "cash", "shoe"]
FROZEN, V2 = recipe.FROZEN_VERSION, recipe.V2_VERSION
FROZEN_KW = dict(alternatives=[], trailing_you=False, recipe_version=FROZEN)


def _decode(words, prob=0.9):
    return [{"word": w, "start": 0.0, "end": 0.0, "probability": prob} for w in words]


def _run(transcript, version, word_decode=None, **kw):
    kw.setdefault("recipe_config", "ra_informed")
    kw.setdefault("alternatives", [])
    kw.setdefault("trailing_you", False)
    return classify_tokens(transcript, LIST, recipe_version=version,
                           word_decode=word_decode, **kw)


def _score(transcript, version, word_decode=None, **kw):
    kw.setdefault("recipe_config", "ra_informed")
    kw.setdefault("alternatives", [])
    kw.setdefault("trailing_you", False)
    return score_recall(transcript, LIST, recipe_version=version, word_decode=word_decode, **kw)


def _by_word(tokens, word):
    return [t for t in tokens if t["word_said"] == word]


def test_default_version_is_frozen_and_v2_is_known():
    assert recipe.FROZEN_VERSION == "2026-09-freeze"
    assert recipe.RECIPE_VERSIONS == ("2026-09-freeze", "2026-09-v2")
    assert recipe.resolve_version(None) == recipe.RECIPE_VERSION
    assert recipe.is_v2("2026-09-v2") and not recipe.is_v2("2026-09-freeze")
    with pytest.raises(ValueError):
        recipe.resolve_version("2026-10-nothing")
    # the frozen manifest block is unchanged; v2 adds its own block
    assert recipe.thresholds(FROZEN) == recipe.THRESHOLDS
    assert recipe.thresholds(V2)["v2"] == recipe.THRESHOLDS_V2
    assert "v2" not in recipe.THRESHOLDS
    # the rejected no-anchor fallback is not part of v2
    assert not hasattr(recipe, "INTRUSION_ANCHOR_MIN_CORRECT")
    assert "intrusion_anchor_min_correct" not in recipe.THRESHOLDS_V2


def test_v1_doubled_trailing_segment_is_dropped_under_v2_and_kept_under_frozen():
    text = "dime, log, pear. dime, log, pear."
    decode = _decode(["dime,", "log,", "pear."])           # heard once
    frozen_tokens, _ = _run(text, FROZEN, decode)
    assert [t["classification"] for t in frozen_tokens] == [
        "correct", "correct", "intrusion-new", "repetition", "repetition", "repetition"]
    v2_tokens, _ = _run(text, V2, decode)
    assert [t["classification"] for t in v2_tokens] == [
        "correct", "correct", "intrusion-new", "deleted", "deleted", "deleted"]
    assert [t["position"] for t in v2_tokens] == [1, 2, 3, 4, 5, 6]
    assert all(t["note"] == recipe.LOOP_NOTE for t in v2_tokens[3:])
    s_frozen, s_v2 = _score(text, FROZEN, decode), _score(text, V2, decode)
    assert s_frozen["n_repetitions"] == 3 and s_frozen["n_intrusions"] == 2
    assert s_v2["n_repetitions"] == 0 and s_v2["n_intrusions"] == 1
    assert s_v2["n_correct"] == s_frozen["n_correct"] == 2
    assert s_v2["n_loop_guard_dropped"] == 3
    # the frozen result carries none of the v2 columns
    assert "n_loop_guard_dropped" not in s_frozen
    assert "intrusion_fallback" not in s_v2 and "n_intrusions_fallback" not in s_v2


def test_v1_supported_repetition_is_kept_and_guard_is_inert_without_a_decode():
    text = "dime, log, dime."
    # the word decode heard dime twice: a genuine repetition
    s = _score(text, V2, _decode(["dime,", "log,", "dime."]))
    assert s["n_repetitions"] == 1 and s["n_loop_guard_dropped"] == 0
    # a separated repeat is not a copy of the block before it: never a loop,
    # whatever the decode says
    s = _score(text, V2, _decode(["dime,", "log,"]) + _decode(["dime."], prob=0.0))
    assert s["n_repetitions"] == 1 and s["n_loop_guard_dropped"] == 0
    # an adjacent copy with near-zero probability in the decode is dropped
    decode = _decode(["dime,", "log,"]) + _decode(["log."], prob=0.0)
    s = _score("dime, log, log.", V2, decode)
    assert s["n_repetitions"] == 0 and s["n_loop_guard_dropped"] == 1
    # no word decode at all: nothing is dropped
    s = _score(text, V2, [])
    assert s["n_repetitions"] == 1 and s["n_loop_guard_dropped"] == 0
    # a repeat that is not a verbatim copy of the preceding block is not a loop
    s = _score("dime, log, pen, dime.", V2, _decode(["dime,", "log,", "pen,"]))
    assert s["n_repetitions"] == 1 and s["n_loop_guard_dropped"] == 0


def test_v1_stutter_run_and_text_form():
    text = "stone, seed, seed, seed, seed."
    decode = _decode(["stone,", "seed,"])
    tokens, _ = _run(text, V2, decode)
    seeds = _by_word(tokens, "seed")
    assert seeds[0]["classification"] == "intrusion-new"
    assert [t["classification"] for t in seeds[1:]] == ["deleted"] * 3
    new_text, dropped = recipe.loop_guard_text(text, decode)
    assert new_text == "stone, seed." and dropped == ["seed", "seed", "seed"]
    # the first occurrence of anything is never deleted, punctuation survives
    new_text, dropped = recipe.loop_guard_text("Dime, log. Dime, log.", _decode(["Dime,", "log."]))
    assert new_text == "Dime, log." and dropped == ["dime", "log"]


def test_v1_contractions_are_compared_on_a_harmonised_form():
    assert recipe.loop_compare_form("i'm") == ("i", "am")
    assert recipe.loop_compare_form("dime") == ("dime",)
    # the production decode wrote "i'm" twice; the word decode heard the same
    # two utterances as "I am" and "I'm": both supported, nothing is a loop
    text = "dime, i'm done, i'm done"
    decode = _decode(["dime,", "I", "am", "done,", "I'm", "done"])
    s = _score(text, V2, decode)
    assert s["n_loop_guard_dropped"] == 0
    assert s["n_repetitions"] == 2               # done, done: intrusion plus its repeat (D2)
    # the reverse split: production "i am", word decode "I'm" once and "I am" once
    s = _score("dime, i am done, i am done", V2, _decode(["dime,", "I'm", "done,", "I", "am", "done"]))
    assert s["n_loop_guard_dropped"] == 0
    # with the word decode hearing the clause once, the verbatim copy is a loop
    s = _score(text, V2, _decode(["dime,", "I", "am", "done"]))
    assert s["n_loop_guard_dropped"] == 2 and s["n_repetitions"] == 0
    # a plain word is still compared as itself
    assert recipe.compare_counter(["i'm", "dime", "don't"]) == {"i": 1, "am": 1, "dime": 1,
                                                                 "do": 1, "not": 1}


def test_no_anchor_fallback_is_not_applied_under_v2():
    # nothing credited: the frozen span rule applies under both versions
    text = "apple, pear, um, plum, plum"
    frozen = _score(text, FROZEN, [])
    v2 = _score(text, V2, [])
    assert frozen["n_correct"] == v2["n_correct"] == 0
    assert frozen["n_intrusions"] == v2["n_intrusions"] == 0
    assert frozen["n_intrusions_all"] == v2["n_intrusions_all"] == 3
    tokens, _ = _run(text, V2, [])
    assert _by_word(tokens, "apple")[0]["note"] == "outside recall span"
    assert not any("fallback" in t for t in tokens)


def test_v3_narration_clause_tokens_are_not_counted():
    text = "dime, pear, it's hard to speak when i'm having breakfast, log"
    frozen = _score(text, FROZEN, [])
    assert frozen["n_intrusions"] == 9              # pear plus every word of the clause
    v2 = _score(text, V2, [])
    assert v2["n_intrusions"] == 1 and v2["n_intrusions_narration"] == 8
    assert v2["n_correct"] == frozen["n_correct"] == 2
    assert v2["n_intrusions_all"] == frozen["n_intrusions_all"]     # pre-freeze meaning kept
    tokens, _ = _run(text, V2, [])
    assert _by_word(tokens, "breakfast")[0]["note"] == recipe.NARRATION_NOTE
    assert _by_word(tokens, "breakfast")[0]["classification"] == "intrusion-new"
    assert _by_word(tokens, "pear")[0]["note"] == ""
    # a clause with a list word, or without a pronoun/auxiliary, or too short, is not narration
    assert recipe.is_narration_clause(["river", "house", "home", "coin"], LIST) is False
    assert recipe.is_narration_clause(["i", "am", "not", "sure", "about", "dime"], LIST) is False
    assert recipe.is_narration_clause(["i", "got", "nothing"], LIST) is False
    assert recipe.is_narration_clause(["those", "are", "the", "only", "ones", "i", "remember"], LIST)


def test_v3_runs_after_credits_so_a_near_miss_or_rescue_clause_is_never_narration():
    # "dine" is a near miss for "dime" (R4, first three content positions):
    # the clause text alone holds no list word, but the credited token makes
    # the clause a recall clause and its other intrusion tokens stay counted
    text = "dine i think that was one, log"
    tokens, _ = _run(text, V2, [])
    dine = _by_word(tokens, "dine")[0]
    assert dine["classification"] == "correct" and dine["matched_list_word"] == "dime"
    s = _score(text, V2, [])
    assert s["n_correct"] == 2 and s["n_intrusions_narration"] == 0
    assert s["n_intrusions"] == _score(text, FROZEN, [])["n_intrusions"]
    # an R3 rescue credit inside the clause has the same effect
    text = "pen, i think it was a dyke i guess, log"
    alt = [{"word": "dyke", "alt2": "duck", "alt2_prob": 0.4, "alt3": "", "alt3_prob": 0.0}]
    rescued = _score(text, V2, [], alternatives=alt)
    plain = _score(text, V2, [])
    assert plain["n_intrusions_narration"] > 0
    if rescued["n_correct"] == plain["n_correct"] + 1:      # the rescue fired
        assert rescued["n_intrusions_narration"] == 0
        assert _by_word(_run(text, V2, [], alternatives=alt)[0], "dyke")[0]["matched_list_word"] == "duck"


def test_frozen_outputs_are_unchanged_by_the_switch():
    cases = [
        ("dime, log, pear. dime, log, pear.", _decode(["dime,", "log,", "pear."])),
        ("apple, pear, um, plum, plum", []),
        ("dime, pear, it's hard to speak when i'm having breakfast, log", []),
        ("gym, dime, did I say gym already, log", None),
    ]
    for text, decode in cases:
        # the frozen version ignores the word decode entirely
        a = score_recall(text, LIST, **FROZEN_KW)
        b = score_recall(text, LIST, **FROZEN_KW, word_decode=decode)
        assert a == b
        ta, _ = classify_tokens(text, LIST, **FROZEN_KW)
        tb, _ = classify_tokens(text, LIST, **FROZEN_KW, word_decode=decode)
        assert ta == tb
        assert not any("narration" in t or "fallback" in t for t in ta)
        assert not any(k in a for k in ("n_loop_guard_dropped", "n_intrusions_narration"))
        # and, when the environment selects the frozen default, the default is that version
        if recipe.RECIPE_VERSION == FROZEN:
            assert score_recall(text, LIST, alternatives=[], trailing_you=False) == a
