"""Romanised Hindi ("Hinglish"), romanised Telugu, and Devanagari/Telugu-script symptom words, glossed with their English
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
# Romanised Telugu ("naaku jvaram undi" = "I have a fever"). Hyderabad is Telugu
# country, so this is the other spelling customers will plausibly type.
_GLOSSARY += [
    (r"jv?waram|jvaram|jwaram", "fever"),
    (r"podi\s+dag+u", "dry"),
    (r"dag+u", "cough"),
    (r"jalubu", "cold"),
    (r"th?ala\s*nop+i", "headache"),
    (r"gontu\s*nop+i", "sore throat"),
    (r"tum+u|tum+ulu", "sneezing"),
    (r"mukku\s+(?:kaar|karr?)\w*", "runny nose"),
    (r"pil+alu|pil+ala|pil+avadu|pil+aki|pil+a\b|bidda\w*", "child"),
    (r"kapham|kaphamu", "wet cough with phlegm"),
]
_COMPILED = [(re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE), gloss) for pattern, gloss in _GLOSSARY]

# Devanagari and Telugu SCRIPT. Plain substring matching: \b is unusable here
# because the vowel signs inside these words are not "word" characters to the
# re module, so a word boundary would fire in the middle of a word.
_SCRIPT_GLOSSARY = [
    (("बुखार", "ज्वर"), "fever"),
    (("सिरदर्द", "सर दर्द", "सिर दर्द"), "headache"),
    (("बदन दर्द", "शरीर में दर्द", "बदन में दर्द"), "body ache"),
    (("सूखी खांसी", "सूखी खाँसी", "सूखा खांसी"), "dry"),
    (("बलगम",), "wet cough with phlegm"),
    (("खांसी", "खाँसी"), "cough"),
    (("सर्दी", "जुकाम", "ज़ुकाम", "नज़ला", "नजला"), "cold"),
    (("गले में दर्द", "गला खराब", "गले में खराश"), "sore throat"),
    (("छींक",), "sneezing"),
    (("नाक बह",), "runny nose"),
    (("बच्चा", "बच्ची", "बच्चे", "बच्चों", "बेटा", "बेटी"), "child"),
    (("జ్వరం", "జ్వరము"), "fever"),
    (("తలనొప్పి", "తల నొప్పి"), "headache"),
    (("ఒళ్ళు నొప్పులు", "ఒళ్లు నొప్పులు", "ఒంటి నొప్పులు"), "body ache"),
    (("పొడి దగ్గు",), "dry"),
    (("కఫం",), "wet cough with phlegm"),
    (("దగ్గు",), "cough"),
    (("జలుబు",), "cold"),
    (("గొంతు నొప్పి", "గొంతునొప్పి"), "sore throat"),
    (("తుమ్ము", "తుమ్ములు"), "sneezing"),
    (("ముక్కు కారడం", "ముక్కు కారుతోంది"), "runny nose"),
    (("పిల్లవాడు", "పిల్లలు", "పిల్ల", "బిడ్డ", "చిన్న పిల్ల"), "child"),
]



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
    for words, english in _SCRIPT_GLOSSARY:
        if any(word in text for word in words) and english not in added and not re.search(rf"\b{re.escape(english)}\b", lower):
            added.append(english)
    if not added:
        return text
    return f"{text} ({'; '.join(added)})"
