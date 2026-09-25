"""Synthetic-transcript tests for the frozen recipe (2026-09-freeze).

Every recall rule R2 to R10 is exercised on made-up transcripts against made-up
word lists; no recording, participant or hand score is used. The alternatives
table (R3) and the word-level verdict (R6) are injected through the keyword
arguments classify_tokens exposes for exactly this purpose.
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


def _by_word(tokens, word):
    return [t for t in tokens if t["word_said"] == word]


def _classes(tokens):
    return [(t["word_said"], t["classification"]) for t in tokens]


def _frozen(transcript, targets=LIST, **kw):
    kw.setdefault("recipe_config", "ra_informed")
    kw.setdefault("alternatives", [])
    kw.setdefault("trailing_you", False)
    return classify_tokens(transcript, targets, **kw)


def test_recipe_version_and_thresholds_are_frozen():
    assert recipe.FROZEN_VERSION == "2026-09-freeze"
    assert recipe.resolve_version("2026-09-freeze") == recipe.FROZEN_VERSION
    # The module default is the frozen version unless the exploratory switch is
    # set in the environment; the v2 suite runs with it set, so only check the
    # default when it is not.
    if not os.environ.get("SWC_RECIPE_VERSION"):
        assert recipe.RECIPE_VERSION == recipe.FROZEN_VERSION
    assert recipe.RESCUE_CHOSEN_PROB_MAX == 0.50
    assert recipe.RESCUE_ENTROPY_MAX == 5.0
    assert recipe.RESCUE_ALT_PROB_MIN == 0.05
    assert recipe.NEAR_MISS_MAX_POSITION == 3
    assert recipe.COMPOUND_MIN_LEN == 6
    assert recipe.TRAILING_YOU_PROB_MAX == 0.15
    assert recipe.SPAN_STOP_DEFAULT in recipe.SPAN_STOP_VARIANTS
    assert "still" not in recipe.HOMO


def test_legacy_scorer_is_unchanged():
    tokens, never = classify_tokens("Jim, dime, dime, apple, um, that's it", LIST, legacy=True)
    assert _classes(tokens) == [("jim", "intrusion-new"), ("dime", "correct"),
                                ("dime", "repetition"), ("apple", "intrusion-new"),
                                ("um", "deleted"), ("that's", "deleted"), ("it", "deleted")]
    assert "gym" in never and "dime" not in never


def test_r2_confusion_map_is_list_gated_and_config_gated():
    tokens, _ = _frozen("Jim, dime")
    jim = _by_word(tokens, "jim")[0]
    assert jim["classification"] == "correct" and jim["matched_list_word"] == "gym"
    assert jim["note"] == "confusion map: jim -> gym"
    # same word, a list without gym: the map must not fire
    tokens, _ = _frozen("Jim, dime", targets=["dime", "log"])
    assert _by_word(tokens, "jim")[0]["classification"] == "intrusion-new"
    # automatic configuration disables the map only
    tokens, _ = _frozen("Jim, dime", recipe_config="automatic")
    assert _by_word(tokens, "jim")[0]["classification"] == "intrusion-new"


def test_r2_lone_i_and_thank_you_guard():
    tokens, _ = _frozen("Dime. I. Thank you. Thank, log")
    eye = _by_word(tokens, "i")[0]
    assert eye["matched_list_word"] == "eye" and eye["classification"] == "correct"
    thanks = _by_word(tokens, "thank")
    assert thanks[0]["classification"] == "intrusion-new"        # "thank you" never converts
    assert thanks[1]["matched_list_word"] == "bank"               # bare "thank" does
    # "I think dime": the pronoun inside a clause stays a filler
    tokens, _ = _frozen("I think dime")
    assert _by_word(tokens, "i")[0]["classification"] == "deleted"


def test_r3_rescue_from_alternatives():
    # "pinch" is two edits from "pen", so the near-miss rule (R4) cannot claim it
    alts = [{"word": "Pinch", "chosen_prob": 0.30, "entropy": 2.0,
             "alt2": "pen (0.20)", "alt3": "pan (0.10)"}]
    tokens, never = _frozen("pinch, dime", alternatives=alts)
    pin = _by_word(tokens, "pinch")[0]
    assert pin["classification"] == "correct" and pin["matched_list_word"] == "pen"
    assert pin["note"] == "rescue: Pinch -> pen"
    assert "pen" not in never
    # gates: confident written word, flat distribution, weak candidate
    for row in [dict(alts[0], chosen_prob=0.60), dict(alts[0], entropy=5.5),
                dict(alts[0], alt2="pen (0.04)", alt3="")]:
        tokens, _ = _frozen("pinch, dime", alternatives=[row])
        assert _by_word(tokens, "pinch")[0]["classification"] == "intrusion-new"
    # one credit per list word: an already credited word is not rescued twice
    tokens, _ = _frozen("pen, pinch, dime", alternatives=alts)
    assert _by_word(tokens, "pinch")[0]["classification"] == "intrusion-new"
    # no alternatives rows means no rescue
    tokens, _ = _frozen("pinch, dime", alternatives=[])
    assert _by_word(tokens, "pinch")[0]["classification"] == "intrusion-new"


def test_r4_near_miss_only_in_first_three_positions():
    tokens, _ = _frozen("dine, log, duck, dolt")
    dine = _by_word(tokens, "dine")[0]
    assert dine["matched_list_word"] == "dime" and dine["note"] == "near-miss (pos 1): dine -> dime"
    assert _by_word(tokens, "dolt")[0]["classification"] == "intrusion-new"  # position 4
    # already credited words are not forced onto; ties break alphabetically
    tokens, _ = _frozen("dime, dine, log")
    assert _by_word(tokens, "dine")[0]["classification"] == "intrusion-new"
    tokens, _ = _frozen("cask, log", targets=["cash", "cast", "log"])
    assert _by_word(tokens, "cask")[0]["matched_list_word"] == "cash"


def test_r5_compound_split_and_extension():
    tokens, never = _frozen("winstone, dime")
    halves = _by_word(tokens, "winstone")
    assert [h["matched_list_word"] for h in halves] == ["stone", "wind"] or \
        [h["matched_list_word"] for h in halves] == ["wind", "stone"]
    assert all(h["classification"] == "correct" for h in halves)
    assert "wind" not in never and "stone" not in never
    # exact merge of a studied word and a word from another CAT list
    other = sorted(recipe.all_list_words() - set(LIST))
    if other:
        o = next((w for w in other if len(w) >= 3), other[0])
        tokens, _ = _frozen(f"log{o}, dime")
        rows = _by_word(tokens, f"log{o}")
        assert rows[0]["matched_list_word"] == "log" and rows[0]["classification"] == "correct"
        assert rows[1]["classification"] == "intrusion-new"
    # short tokens never split
    tokens, _ = _frozen("penlog, dime", targets=["pen", "log", "dime"])
    assert _by_word(tokens, "penlog")[0]["note"].startswith("compound split")  # exact, main loop
    tokens, _ = _frozen("penlo, dime", targets=["pen", "log", "dime"])
    assert _by_word(tokens, "penlo")[0]["classification"] == "intrusion-new"


def test_r6_trailing_you_artifact():
    tokens, _ = _frozen("dime, log, you", trailing_you=True)
    assert tokens[-1]["classification"] == "deleted" and tokens[-1]["note"] == "recognizer artifact"
    tokens, _ = _frozen("dime, log, you", trailing_you=False)
    assert tokens[-1]["classification"] == "intrusion-new"


def test_r7_retraction_and_giveup():
    tokens, _ = _frozen("dime, no, log")
    assert _by_word(tokens, "dime")[0]["classification"] == "deleted"
    assert _by_word(tokens, "dime")[0]["note"] == "retracted"
    assert _by_word(tokens, "no")[0]["classification"] == "deleted"
    tokens, _ = _frozen("dime, no I can't remember")
    assert _by_word(tokens, "dime")[0]["classification"] == "correct"
    assert _by_word(tokens, "no")[0]["note"] == "no: opens a give-up phrase"
    # an intrusion followed by "no" is retracted too, and "no" is never an intrusion
    s = score_recall("apple, no, dime", LIST, alternatives=[], trailing_you=False)
    assert s["n_intrusions"] == 0 and s["n_intrusions_all"] == 0 and s["n_correct"] == 1


def test_r8_self_correction_reponly():
    tokens, _ = _frozen("gym, dime, did I say gym already, log")
    gyms = _by_word(tokens, "gym")
    assert gyms[0]["classification"] == "correct"
    assert gyms[1]["classification"] == "deleted"
    assert gyms[1]["note"] == "self-correction (repeated word named)"
    for w in ("did", "say", "already"):
        assert _by_word(tokens, w)[0]["note"] == "self-correction"
    # the named word is exempt only if it is currently a repetition
    tokens, _ = _frozen("dime, did I say gym already, log")
    assert _by_word(tokens, "gym")[0]["classification"] == "correct"
    # generic statement claims the preceding content token
    tokens, _ = _frozen("gym, dime, gym, I already said that, log")
    assert _by_word(tokens, "gym")[1]["classification"] == "deleted"


def test_r9_recall_span_and_r10_repeated_intrusion():
    text = "okay so apple, dime, pear, log, um, plum, plum"
    tokens, _ = _frozen(text, span_stop="fillers")
    assert _by_word(tokens, "apple")[0]["note"] == "outside recall span"   # before first list word
    assert _by_word(tokens, "pear")[0]["in_span"]                          # between list words
    plums = _by_word(tokens, "plum")
    assert plums[0]["note"] == "outside recall span"                       # after a filler
    assert plums[1]["classification"] == "repetition"
    assert plums[1]["note"] == "repeated intrusion; outside recall span"
    s = score_recall(text, LIST, alternatives=[], trailing_you=False, span_stop="fillers")
    assert s["n_correct"] == 2
    assert s["n_intrusions"] == 1            # pear only
    assert s["n_intrusions_all"] == 3        # apple, pear, plum (old convention)
    assert s["n_repetitions"] == 1 and s["n_repeated_intrusions"] == 1
    # a repeated intrusion inside the span counts as both (decision D2)
    s = score_recall("dime, plum, plum, log", LIST, alternatives=[], trailing_you=False)
    assert s["n_repetitions"] == 1 and s["n_intrusions"] == 2 and s["n_intrusions_in_span"] == 1
    # chatter variant stops the trailing run at a chatter word
    text = "dime, log, pear, yeah, plum"
    fillers = score_recall(text, LIST, alternatives=[], trailing_you=False, span_stop="fillers")
    chatter = score_recall(text, LIST, alternatives=[], trailing_you=False, span_stop="chatter")
    assert fillers["n_intrusions"] == 3 and chatter["n_intrusions"] == 1


def test_configuration_validation():
    with pytest.raises(ValueError):
        classify_tokens("dime", LIST, recipe_config="manual", alternatives=[], trailing_you=False)
    with pytest.raises(ValueError):
        classify_tokens("dime", LIST, span_stop="none", alternatives=[], trailing_you=False)


def test_ed1():
    assert recipe.ed1("dine", "dime") and recipe.ed1("wog", "log")
    assert recipe.ed1("stone", "stones") and recipe.ed1("pen", "pe")
    assert not recipe.ed1("dime", "dime") and not recipe.ed1("dime", "dome2")
    assert recipe.within_one_edit("keystone", "keystone")
