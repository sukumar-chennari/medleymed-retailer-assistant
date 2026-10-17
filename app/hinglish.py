"""Romanised Hindi ("Hinglish") symptom words, glossed with their English
meaning. The shop's customers write like this — "mujhe bukhar hai", "mera
bachha ko khansi hai" — and every one of those was declined as out of scope
(found Oct 17), because the symptom vocabulary is English-only. Rather than
duplicate every keyword list, the original words are kept and the English
meaning is appended in parentheses ("mujhe bukhar hai (fever)"), so each
existing branch — symptom classification, the child-or-adult and dry-or-wet
questions, the red-flag check — sees words it already understands.
Dependency-free on purpose. Replies stay in English."""

import re

# (pattern, English gloss). Word-boundary matches on lowercase text; spellings
# vary a lot in romanised Hindi, so the common ones are listed.
_GLOSSARY = [
    (r"bukh[a]{1,2}r|bukhar", "fever"),
    (r"sir\s*dard|sar\s*dard|sirdard|sardard", "headache"),
    (r"badan\s+(?:mein\s+|me\s+)?dard|body\s*dard", "body ache"),
    (r"khan?si|kh[a]{2}nsi|khasi", "cough"),
    (r"sardi|zukam|zukaam|jukam|jukaam|nazla", "cold"),
    (r"naak\s+(?:beh|bah)|nak\s+(?:beh|bah)", "runny nose"),
    (r"gal[ae]\s+(?:mein\s+|me\s+)?(?:dard|kharash|kharab)|gala\s+(?:dard|kharab)", "sore throat"),
    (r"chh?ee?nk(?:e|en|ein)?|chinke", "sneezing"),
    (r"bac+h+c*h*[aeiou]|bachon|bacho", "child"),
    (r"(?:khud|apne)\s+ke\s+liye|apne\s+aap\s+ke\s+liye|mere\s+liye|apne\s+liye", "for myself"),
    (r"suk+hi\s+khan?si|sook+hi\s+khan?si", "dry"),
    (r"balgam|geeli\s+khan?si|gili\s+khan?si|kapha", "wet cough with phlegm"),
]
_COMPILED = [(re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE), gloss) for pattern, gloss in _GLOSSARY]


def gloss(text: str) -> str:
    """The text with English glosses appended for any Hinglish words in it
    (unchanged when there are none, or the English word is already there)."""
    if not text:
        return text
    lower = text.lower()
    added = []
    for pattern, english in _COMPILED:
        if pattern.search(text) and english not in added and not re.search(rf"\b{re.escape(english)}\b", lower):
            added.append(english)
    if not added:
        return text
    return f"{text} ({'; '.join(added)})"
