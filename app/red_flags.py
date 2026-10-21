"""Messages that describe a medical emergency must never reach the ordinary
fever/cold flow. Found Oct 17 by probing the live route:
  - "I took 10 paracetamol tablets" (an overdose) was answered with "Is this
    fever for a child, or for an adult/yourself?";
  - "my baby is 2 months old and has a fever" was answered with a product
    recommendation ("Children's Paracetamol Syrup would be a good fit") when a
    fever under 3 months needs a doctor now, not a medicine;
  - "diagnose my chest pain" got the generic out-of-scope decline.
So these are caught in code, first, with a fixed reply that sends the person
to emergency care. Dependency-free on purpose. The bias is toward escalating:
a flag is dropped only when EVERY mention of it is negated ("no chest pain"),
and nothing here ever recommends or orders a product.
"""

import re

EMERGENCY_NUMBERS = "112 in India, 911 in the US"

OVERDOSE_REPLY = (
    "This could be a medicine overdose or poisoning, and it shouldn't wait for symptoms to appear. "
    f"Please call your local emergency number now ({EMERGENCY_NUMBERS}) or your poison control centre, "
    "and tell them what was taken, how much, and when. If the person is drowsy, confused, vomiting, "
    "having trouble breathing or having a seizure, call emergency services immediately. "
    "I can't help further with this here."
)
EMERGENCY_REPLY = (
    "What you describe can be serious, and I can't assess it here. Please get emergency medical care now — "
    f"call your local emergency number ({EMERGENCY_NUMBERS}) or go to the nearest emergency department. "
    "Don't wait to see if it passes."
)
INFANT_FEVER_REPLY = (
    "A fever in a baby under 3 months can be serious. Please contact a doctor or go to urgent care right away "
    "rather than giving any medicine — I can't recommend a product for this. "
    f"If the baby is unusually sleepy, feeding poorly or struggling to breathe, call emergency services ({EMERGENCY_NUMBERS})."
)

_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|nor|neither|isn't|aren't|don't|doesn't|didn't|hasn't|haven't)\b|n't\b", re.IGNORECASE)
_NEGATION_LOOKBACK = 25

# Tablets in one go before it counts as an overdose on its own. The highest
# daily total on any product we carry is 8 x 500 mg paracetamol; the message
# rarely says whether it was one sitting, so from 8 up it is escalated.
_OVERDOSE_TABLET_THRESHOLD = 8

_MEDICINE_WORDS = (
    r"(?:tablets?|pills?|capsules?|syrup|bottle|bottles|pack|packs|strips?|doses?|medicine|medication|meds|"
    r"paracetamol|acetaminophen|ibuprofen|cetirizine|pseudoephedrine|guaifenesin|dextromethorphan|antihistamine)"
)
_TAKER = r"(?:i|he|she|they|we|my\s+\w+|the\s+(?:child|kid|baby|toddler))"
_TOOK = r"(?:took|taken|swallowed|ate|eaten|drank|had|downed|popped)"

_OVERDOSE_RES = [
    re.compile(
        rf"\b{_TAKER}\s+(?:just\s+|accidentally\s+|probably\s+)?(?:have\s+|has\s+|may have\s+|think\s+i\s+)?overdos(?:e|ed|ing)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i|we)\s+(?:think|believe|am afraid)\s+(?:i|we|he|she|they)\s+overdosed\b", re.IGNORECASE),
    re.compile(rf"\b{_TOOK}\s+too\s+(?:many|much)\b", re.IGNORECASE),
    re.compile(rf"\b{_TOOK}\s+(?:a|the|an)\s+(?:whole|entire)\s+(?:bottle|pack|strip|box)\b", re.IGNORECASE),
    re.compile(rf"\b{_TOOK}\s+(?:all|the whole lot of)\s+(?:the|of the|my|his|her)\s+{_MEDICINE_WORDS}\b", re.IGNORECASE),
    re.compile(rf"\b{_TOOK}\s+(?:double|twice|triple|three times)\s+(?:the\s+)?(?:dose|amount|recommended)\b", re.IGNORECASE),
    re.compile(r"\b(?:took|taken)\s+more\s+than\s+(?:the\s+)?(?:recommended|prescribed|directed|label|maximum|max)\b", re.IGNORECASE),
    re.compile(rf"\baccidentally\s+{_TOOK}\b.{{0,40}}{_MEDICINE_WORDS}", re.IGNORECASE),
    re.compile(
        rf"\b(?:my|the)\s+(?:child|kid|baby|toddler|son|daughter|\w+[- ]year[- ]old)\s+(?:swallowed|ate|drank|got into)\b.{{0,40}}{_MEDICINE_WORDS}",
        re.IGNORECASE,
    ),
]
_TABLET_COUNT_RE = re.compile(
    rf"\b{_TOOK}\s+(?:about\s+|around\s+|like\s+|roughly\s+)?(\d{{1,3}})\s+(?:\w+\s+){{0,3}}{_MEDICINE_WORDS}\b",
    re.IGNORECASE,
)

_EMERGENCY_RES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bchest\s+(?:pain|tightness|pressure)\b",
        r"\b(?:trouble|difficulty|struggling|hard)\s+(?:to\s+)?breath\w*\b",
        r"\b(?:can'?t|cannot|unable\s+to)\s+breathe?\b",
        r"\bshort(?:ness)?\s+of\s+breath\b",
        r"\bblue\s+(?:lips|face|skin)\b",
        r"\b(?:seizure|seizures|convulsion|convulsions|convulsing|fitting)\b",
        r"\b(?:unconscious|unresponsive|passed\s+out|fainted|not\s+waking)\b",
        r"\b(?:face\s+droop\w*|slurred\s+speech|sudden\s+(?:weakness|numbness|confusion))\b",
        r"\b(?:anaphyla\w*|swollen\s+(?:face|lips|tongue|throat)|throat\s+(?:is\s+)?(?:closing|swelling)"
        r"|(?:face|lips|tongue|throat)\s+(?:is|are|feels?|got|getting)\s+(?:swollen|swelling))\b",
        r"\b(?:coughing|vomiting|throwing\s+up|spitting)\s+(?:up\s+)?blood\b",
        r"\bstiff\s+neck\b",
        r"\bsuicid\w*|\bwant\s+to\s+(?:die|kill\s+myself)\b",
    )
]

_FEVER_RE = re.compile(r"\b(?:fever|feverish|temperature|temp|hot\s+to\s+the\s+touch|burning\s+up)\b", re.IGNORECASE)
_NEWBORN_RE = re.compile(r"\bnewborn\b", re.IGNORECASE)
_AGE_RE = re.compile(
    r"\b(\d{1,2}|one|two|three|four|a|an)\s*[- ]?\s*(day|days|week|weeks|month|months)[- ]?\s*(?:old)?\b",
    re.IGNORECASE,
)
_WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4}


def _has_unnegated_match(pattern: re.Pattern, text: str) -> bool:
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
        if not _NEGATION_RE.search(before):
            return True
    return False


def is_overdose(text: str) -> bool:
    if any(_has_unnegated_match(p, text) for p in _OVERDOSE_RES):
        return True
    for match in _TABLET_COUNT_RE.finditer(text):
        before = text[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
        if int(match.group(1)) >= _OVERDOSE_TABLET_THRESHOLD and not _NEGATION_RE.search(before):
            return True
    return False


def is_emergency(text: str) -> bool:
    return any(_has_unnegated_match(p, text) for p in _EMERGENCY_RES)


def is_infant_fever(text: str) -> bool:
    """A fever mentioned for a baby under 3 months (newborn, days/weeks old,
    or 1-2 months old)."""
    if not _FEVER_RE.search(text):
        return False
    if _NEWBORN_RE.search(text):
        return True
    for match in _AGE_RE.finditer(text):
        raw, unit = match.group(1).lower(), match.group(2).lower()
        number = int(raw) if raw.isdigit() else _WORD_NUMBERS.get(raw)
        if number is None:
            continue
        if unit.startswith(("day", "week")):
            # "3 days" alone is a duration ("fever for 3 days"), not an age,
            # unless the text says old.
            if re.search(r"old", match.group(0), re.IGNORECASE) and (unit.startswith("day") or number <= 12):
                return True
        elif unit.startswith("month") and number < 3:
            return True
    return False


def check(text: str) -> str | None:
    """The fixed reply for a message describing an emergency, or None."""
    if not text:
        return None
    if is_overdose(text):
        return OVERDOSE_REPLY
    if is_infant_fever(text):
        return INFANT_FEVER_REPLY
    if is_emergency(text):
        return EMERGENCY_REPLY
    return None


# A high or long-lasting fever is not an emergency on its own, but it is not
# something to treat with a product list and nothing else either. Found Oct 17:
# "I have fever 104 for 5 days" and "my fever has lasted a week" got the usual
# adult list with no mention of seeing a doctor. The note is appended to a
# recommendation (see main._add_fever_advice); it never blocks one.
DOCTOR_NOTE = (
    "Please note: a fever this high or this long should be checked by a doctor soon — over-the-counter medicine "
    "alone isn't enough. If there is trouble breathing, confusion, a stiff neck, a rash that doesn't fade, or "
    f"the fever keeps climbing, get emergency care ({EMERGENCY_NUMBERS})."
)

_HIGH_FEVER_F = 103.0
_HIGH_FEVER_C = 39.4
_TEMP_RE = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d)?)\s*(?:°|degrees?|deg)?\s*(f\b|c\b|fahrenheit|celsius)?", re.IGNORECASE)
_NUMBER_WORDS = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_DURATION_RE = re.compile(
    r"\b(?:for|since|past|last(?:ed|ing)?|over|about|almost|nearly)\s+(?:the\s+)?(?:last\s+|past\s+)?"
    r"(\d{1,2}|three|four|five|six|seven|eight|nine|ten|a|one|two|several|many)\s*(days?|weeks?|week)\b",
    re.IGNORECASE,
)
_LASTED_A_WEEK_RE = re.compile(r"\b(?:a|one|1)\s+week\b|\bweeks?\b", re.IGNORECASE)
_LONG_DAYS = 3


def _high_temperature(text: str) -> bool:
    """With a unit the range is unambiguous. Without one only values that can
    not be an age or a count are used: a decimal 39.4-43 (Celsius) or a whole
    103-110 (Fahrenheit) — "I'm 40" must not read as a 40 C fever."""
    for match in _TEMP_RE.finditer(text):
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        if unit.startswith("c"):
            if _HIGH_FEVER_C <= value <= 43:
                return True
        elif unit.startswith("f"):
            if _HIGH_FEVER_F <= value <= 110:
                return True
        elif "." in match.group(1):
            if _HIGH_FEVER_C <= value <= 43:
                return True
        elif _HIGH_FEVER_F <= value <= 110:
            return True
    return False


def _long_fever(text: str) -> bool:
    for match in _DURATION_RE.finditer(text):
        raw, unit = match.group(1).lower(), match.group(2).lower()
        if unit.startswith("week"):
            return True  # any number of weeks is well past 3 days
        number = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw) or (_LONG_DAYS if raw in {"several", "many"} else None)
        if number is not None and number >= _LONG_DAYS:
            return True
    return False


def doctor_note(text: str) -> str | None:
    """DOCTOR_NOTE when the text describes a fever at or above 103 F / 39.4 C
    or one that has lasted 3 days or more (not negated), else None."""
    if not text or not _FEVER_RE.search(text):
        return None
    if _high_temperature(text) or _long_fever(text):
        return DOCTOR_NOTE
    return None
