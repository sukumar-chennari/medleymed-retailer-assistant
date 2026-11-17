"""The dashboard's product search (static/app.js filterCatalog), run for real under Node.

There is no browser test runner in this project, but these two functions are plain
JavaScript, so they are extracted from the page's own source and executed. Skipped when
Node is not installed (the GitHub runners have it)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app import store

APP_JS = (Path(__file__).resolve().parent.parent / "static" / "app.js").read_text()

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _function(name: str) -> str:
    match = re.search(rf"function {name}\(.*?\n\}}\n", APP_JS, re.S)
    assert match, f"{name} not found in static/app.js"
    return match.group(0)


def _search(query: str) -> list[str]:
    code = (
        f"const allProducts = {json.dumps(store.get_catalog())};\n"
        f"{_function('searchForm') if 'function searchForm' in APP_JS else ''}\n"
        f"{_function('filterCatalog')}\n"
        f"console.log(JSON.stringify(filterCatalog({json.dumps(query)}).map(p => p.id)));\n"
    )
    result = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_curly_apostrophe_finds_the_same_products_as_a_straight_one():
    # Real bug (Nov 17): "children’s" found nothing, "children's" found two.
    straight = _search("children's")
    assert straight == ["fev-004", "col-006"]
    assert _search("children’s") == straight
    assert _search("Children‘s syrup") == _search("children's syrup")


def test_extra_spaces_do_not_break_a_match():
    assert _search("paracetamol   500mg") == _search("paracetamol 500mg") != []
    assert _search("  PARACETAMOL  ") == _search("paracetamol")


def test_an_empty_search_shows_everything_and_nonsense_shows_nothing():
    assert len(_search("")) == len(store.get_catalog()) == len(_search("   "))
    assert _search("xyz") == []


@pytest.mark.parametrize("query", ["(", "[", "\\", ".*", "$^"])
def test_regex_characters_are_plain_text(query):
    # no exception, and "(" really matches the one name that contains it
    result = _search(query)
    assert result == (["col-004"] if query == "(" else [])
