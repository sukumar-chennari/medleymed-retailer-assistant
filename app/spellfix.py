"""Conservative spelling normalisation for the words the deterministic handlers
depend on. Found Oct 31/Nov 1: "how much is ibuprofin" and "order ibuprofin
200mg tablets" got the child-or-adult question; "ordr Ibuprofen 200mg Tablets"
and "order Ibuprofen 200 mg Tablets" were not recognised; and — the one that
matters — "can i take ibuprofen and paracetmol together" got the model's own
"Yes, you can take ibuprofen and acetaminophen..." because the misspelling
defeated the combination-warning logic, which is the opposite of what the
knowledge base says for paracetamol products.

Deliberately narrow so it cannot turn ordinary words into drug names:
- only tokens of 7+ letters are corrected, and only toward the catalog's own
  drug words and the brand names we know (difflib ratio >= 0.86);
- "200 mg" / "10 ml" are joined; a few command-word typos ("ordr", "oder",
  "gimme") are fixed from a fixed list;
- text containing an "@" is never touched (an email), and main.py does not call
  this for an address reply.
Dependency-light: the vocabulary is built from the catalog.
"""

import difflib
import re

from app import brands, store

_CUTOFF = 0.86
_MIN_LENGTH = 7
_TOKEN_RE = re.compile(r"[A-Za-z]{%d,}" % _MIN_LENGTH)
_UNIT_RE = re.compile(r"(\d)\s+(mg|mcg|ml)\b", re.IGNORECASE)
_COMMAND_TYPOS = {
    "ordr": "order", "oder": "order", "ordar": "order", "odrer": "order", "ordre": "order",
    "gimme": "give me", "plz": "please", "pls": "please",
}
_COMMAND_RE = re.compile(r"\b(" + "|".join(_COMMAND_TYPOS) + r")\b", re.IGNORECASE)


def _vocabulary() -> list[str]:
    words = set()
    for product in store.get_catalog():
        for part in (product["name"], product["active_ingredient"]):
            words |= {w for w in re.findall(r"[a-z]+", part.lower()) if len(w) >= _MIN_LENGTH}
    words -= {"symptom"}   # from "Multi-Symptom": a generic word, not a drug
    words |= {b for b in brands.BRAND_INGREDIENT if len(b) >= _MIN_LENGTH}
    words |= {"acetaminophen"}
    return sorted(words)


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def correct(text: str) -> str:
    if not text or "@" in text:
        return text
    text = _UNIT_RE.sub(r"\1\2", text)
    text = _COMMAND_RE.sub(lambda m: _match_case(m.group(1), _COMMAND_TYPOS[m.group(1).lower()]), text)
    vocabulary = _vocabulary()
    known = set(vocabulary)

    def fix(match: re.Match) -> str:
        token = match.group(0)
        lower = token.lower()
        # a known word, or its plural ("antihistamines"), is already spelled right
        if lower in known or (lower.endswith("s") and lower[:-1] in known):
            return token
        close = difflib.get_close_matches(lower, vocabulary, n=1, cutoff=_CUTOFF)
        return _match_case(token, close[0]) if close else token

    return _TOKEN_RE.sub(fix, text)
