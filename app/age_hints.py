"""Who the medicine is for, when the message already says. Found Oct 26: 14 of
16 messages that state it — "my 3 year old has a fever", "my son has a cold",
"my wife has a fever", "I'm 25 and have a fever", "my 5 year old has a dry
cough" — were still asked "Is this for a child, or for an adult/yourself?",
and the dry-cough one never reached the children's product. As with
hinglish.py, the meaning is appended ("... (child)"), so the existing
clarification, resolution and recommendation code runs unchanged.

Rules, deliberately conservative — when in doubt nothing is appended and the
customer is asked as before:
- an explicit age wins: under 12 -> child, 18 or over -> adult, 12-17 ->
  ambiguous (nothing appended); months/weeks/days old -> child;
- otherwise a relation: son/daughter/boy/girl/newborn/"little one" -> child;
  husband/wife/mother/father/grandparent/elderly -> adult;
- a child and an adult named together ("my son and I both have a cold", "the whole family"), a
  teenager, or a message that already says child/adult/kid/baby/myself gets
  nothing appended.
Dependency-free on purpose."""

import re

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}
_NUM = r"(\d{1,3}|" + "|".join(_WORD_NUMBERS) + r")"
_YEARS_OLD_RE = re.compile(rf"\b{_NUM}[\s-]*(?:years?|yrs?|y/?o)[\s-]*old\b", re.IGNORECASE)
_IM_AGE_RE = re.compile(rf"\b(?:i\s*'?\s*a?m|i\s+am|age[d]?|aged)\s+{_NUM}\b(?!\s*(?:kg|mg|cm|%|ml|days?|weeks?|months?|hours?|minutes?|mins?|miles?|km|blocks?|stops?|away))", re.IGNORECASE)
_SHORT_AGE_RE = re.compile(
    r"\b(\d{1,3})\s*(?:yo|y\.o\.|yr|yrs)\b(?!\s+(?:street|st|road|rd|lane|ln|avenue|ave|nagar|colony|marg))", re.IGNORECASE
)
_INFANT_AGE_RE = re.compile(rf"\b{_NUM}[\s-]*(?:months?|weeks?|days?)[\s-]*old\b", re.IGNORECASE)
_TEEN_RE = re.compile(r"\b(?:teen|teens|teenage|teenager|adolescent)\b", re.IGNORECASE)
# "son-in-law", "boy friend" and "Boy Scouts Street" are not a child (found Nov 27: 14 of 24 adult
# phrasings were classed as children).
_CHILD_RELATION_RE = re.compile(
    r"\b(?:(?:son|daughter)\b(?![\s-]*in[\s-]*law)|boy\b(?!\s*friend|\s+scouts?)|girl\b(?!\s*friend|\s+guides?)|newborn|little\s+one|"
    r"little\s+boy|little\s+girl|grandson|granddaughter|nephew|niece)",
    re.IGNORECASE,
)
# a stated age after the relation: "my nephew is 25", "my daughter is 35", "the girl next door is 25"
_RELATION_AGE_RE = re.compile(
    r"\b(?:son|daughter|nephew|niece|grandson|granddaughter|boy|girl|kid|child)\b(?:\s+[\w'-]+){0,3}?\s+(?:is|was|turned|turns)\s+(\d{1,3})\b"
    r"(?!\s*(?:kg|mg|cm|%|ml|days?|weeks?|months?|hours?|minutes?|mins?|times))",
    re.IGNORECASE,
)
_ADULT_RELATION_RE = re.compile(
    r"\b(?:(?:son|daughter|brother|sister)[\s-]*in[\s-]*law|husband|wife|mother|mom|mum|mama|father|dad|papa|grandmother|grandfather|grandma|grandpa|grandparents?|"
    r"elderly|senior|aunt|uncle|boyfriend|girlfriend)\b",
    re.IGNORECASE,
)
# Already answered: the existing code understands these directly.
_EXPLICIT_RE = re.compile(r"\b(?:child|children|kid|kids|baby|babies|infant|toddler|adult|myself|grown|grownup|grown-up)\b", re.IGNORECASE)
# "my son and I both have a cold": more than one person, so more than one answer.
_MIXED_RE = re.compile(
    r"\b(?:and\s+i|i\s+and|and\s+me|me\s+and|both\s+of\s+us|we\s+both|we\s+all|all\s+of\s+us|whole\s+family|family|us)\b",
    re.IGNORECASE,
)
_ALREADY_GLOSSED_RE = re.compile(r"\((?:child|adult)\)|\b(?:for myself)\b", re.IGNORECASE)
_ADULT_AGE = 18
_CHILD_BELOW = 12


def strip_ages(text: str) -> str:
    """The text with stated ages ("5 year old", "6 months old", "5yo", "I'm 30")
    blanked out, so a number that is an age is not read as a quantity."""
    for pattern in (_YEARS_OLD_RE, _INFANT_AGE_RE, _SHORT_AGE_RE, _IM_AGE_RE):
        text = pattern.sub(" ", text)
    return text


def _number(raw: str) -> int | None:
    raw = raw.lower()
    return int(raw) if raw.isdigit() else _WORD_NUMBERS.get(raw)


def _ages(text: str) -> list[int]:
    ages = []
    for pattern in (_YEARS_OLD_RE, _IM_AGE_RE):
        for match in pattern.finditer(text):
            n = _number(match.group(1))
            if n is not None:
                ages.append(n)
    for match in _SHORT_AGE_RE.finditer(text):
        ages.append(int(match.group(1)))
    for match in _RELATION_AGE_RE.finditer(text):
        ages.append(int(match.group(1)))
    return [a for a in ages if 0 < a < 120]


def age_group(text: str) -> str | None:
    """"child", "adult", or None when the text does not make it clear."""
    if not text or _EXPLICIT_RE.search(text) or _ALREADY_GLOSSED_RE.search(text) or _TEEN_RE.search(text) or _MIXED_RE.search(text):
        return None
    if _INFANT_AGE_RE.search(text):
        return "child"
    ages = _ages(text)
    if ages:
        groups = {"child" if a < _CHILD_BELOW else "adult" if a >= _ADULT_AGE else "teen" for a in ages}
        # A child's age and an adult's age together ("I'm 30, my son is 4") or
        # any 12-17 age: ask as before.
        return groups.pop() if len(groups) == 1 and "teen" not in groups else None
    child, adult = bool(_CHILD_RELATION_RE.search(text)), bool(_ADULT_RELATION_RE.search(text))
    if child and not adult:
        return "child"
    if adult and not child:
        return "adult"
    return None


_MY_CHILD_RE = re.compile(
    r"\bmy\s+(?:\w+\s+){0,2}?(?:son|daughter|grandson|granddaughter|nephew|niece)\b", re.IGNORECASE
)


def states_a_child(text: str) -> bool:
    """True when the text itself says the patient is a child — a stated age under 12
    ("my 5 year old", "6 months old", "5yo") or "my son/daughter" — without needing the
    "(child)" gloss. While an order is pending the gloss is skipped (an address must be
    stored exactly as typed), which left "order ibuprofen for my 5 year old" typed at that
    moment unseen by the child check (found Nov 17). Conservative: any adult age or
    relation in the same text makes it False."""
    if not text or _MIXED_RE.search(text) or _ADULT_RELATION_RE.search(text) or _TEEN_RE.search(text):
        return False
    if _INFANT_AGE_RE.search(text):
        return True
    ages = _ages(text)
    if ages:
        return all(a < _CHILD_BELOW for a in ages)
    return bool(_MY_CHILD_RE.search(text))


def gloss(text: str) -> str:
    group = age_group(text)
    return f"{text} ({group})" if group else text
