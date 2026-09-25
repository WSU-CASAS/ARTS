"""Test setup. The scorers read the vocabulary of every studied list from keys/word_lists.json
(used by the compound rule, R5). That file holds study materials and is not in the repository,
so the tests point it at a small synthetic file instead."""
from __future__ import annotations

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

import config  # noqa: E402
import recipe  # noqa: E402

SYNTHETIC_LISTS = {
    # the list used throughout tests/test_recipe.py and tests/test_recipe_v2.py
    "list_1": ["gym", "eye", "bank", "dime", "wind", "stone", "pen", "log", "duck", "doll", "cash", "shoe"],
    # a second, made-up list: the "other list" words for prior-list intrusions and the R5 extension
    "list_2": ["anchor", "basket", "candle", "desert", "engine", "forest", "garden", "harbor", "island",
               "jacket", "kettle", "lemon"],
}


@pytest.fixture(autouse=True, scope="session")
def synthetic_word_lists(tmp_path_factory):
    path = tmp_path_factory.mktemp("keys") / "word_lists.json"
    path.write_text(json.dumps(SYNTHETIC_LISTS))
    old = config.WORD_LISTS_JSON
    config.WORD_LISTS_JSON = path
    recipe.all_list_words.cache_clear()
    yield
    config.WORD_LISTS_JSON = old
    recipe.all_list_words.cache_clear()
