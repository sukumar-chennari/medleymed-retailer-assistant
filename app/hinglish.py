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
# Shopping words (found Nov 18: "ibuprofen ki kimat" and "mujhe 2 ibuprofen chahiye" reached the
# model, while the same sentences in English are answered in code). Glossed into the English
# the order and price handlers already understand.
_GLOSSARY += [
    (r"daa+m|k[ie]+mat|q[ie]+mat|kitni\s+k[ie]+mat", "price"),
    (r"kitne\s+(?:ka|ke|ki)\s+(?:hai|hain)", "how much is it"),
    (r"chaa?hiy[ea]|chaa?hie", "I want"),
    (r"de\s*do|dedo|dena", "give me"),
    (r"bhej\s*do|bhejo|bhej\s+dena", "send me"),
]
# EMERGENCY words (found Nov 23: 54 of 54 Hindi / Telugu emergency phrasings — "seene mein dard",
# "saans nahi aa rahi", "behosh ho gaya", "dil ka daura", "khoon ki ulti", "marna chahta hoon" —
# were not recognised, so they went into the ordinary fever/cold flow). Glossed into the English
# the red-flag check already understands. Spellings vary a lot, so the common ones are listed.
# "chhati mein jakdan nahi hai" = "no chest tightness": Hindi puts the negation AFTER the symptom.
_NOT_NEGATED = r"(?!\s+(?:\w+\s+)?(?:nahi|nahin|nhi|na)\b)"
_GLOSSARY += [
    (r"(?:(?:seene|seena|chh?aa?ti|chaati|chhaati|gunde|gundelo|chest)\s+(?:mein\s+|me\s+|main\s+|lo\s+)?(?:dard|noppi|nopi)|"
     r"(?:seene|chh?aa?ti)\s+(?:mein\s+)?(?:jakdan|bhaari\s+pan))" + _NOT_NEGATED, "chest pain"),
    (r"saa?ns\s+(?:nahi|na)\s+(?:aa|le|ro)\w*|saa?ns\s+(?:lene\s+mein|lene\s+me|lene\s+main)\s+(?:takleef|taklif|dikkat|problem|pareshani|mushkil)|"
     r"dam\s+ghut\w*|saa?ns\s+phool\w*|oopiri\s+aa?da\w*\s+ledu|oopiri\s+(?:teesuk\w*)\s+kashta\w*", "trouble breathing"),
    (r"be-?hosh\w*|bebosh\w*|spruha\s+ledu|chakkar\s+(?:aa\s*kar|aakar)\s+gir\w*", "unconscious"),
    (r"(?:dil\s+ka\s+(?:daura|dora)|heart\s+attack\s+(?:aa|aaya|aayi|hua)\w*|gunde\s*potu|gunde\s+pot\w*)" + _NOT_NEGATED, "having a heart attack"),
    (r"(?:mirgi|daura\s+pad\w*|jhatke\s+aa\w*|fits?\s+aa\w*)" + _NOT_NEGATED, "seizure"),
    (r"(?:khoon\s+(?:ki|ke)\s+ulti\w*|ulti\s+(?:mein|me|main)\s+khoon|raktham\s+vanthu\w*)" + _NOT_NEGATED, "vomiting blood"),
    (r"khoon\s+(?:band\s+nahi|nahi\s+ruk|beh\s+raha|bah\s+raha)\w*|(?:bahut|zyada|jyada)\s+khoon\b", "severe bleeding"),
    (r"(?:bahut|zyada|jyada|jaada)\s+(?:zyada\s+|jyada\s+)?(?:dawai|dawa|dawaai|goli|goliyan|tablets?)\s+kha\s+li\w*|"
     r"(?:dawai|dawa|dawaai)\s+ki\s+(?:poori\s+|puri\s+)?bottle\s+kha\s+li|(?:goli|goliyan|tablets?)\s+(?:bahut\s+)?(?:zyada|jyada)\s+kha\w*|"
     r"ne\s+(?:dawai|dawa|dawaai|goli|goliyan)\s+(?:kha|nigal)\s+li\w*|tablets?\s+ekkuva\s+vesuk\w*", "took too much medicine"),
    (r"marna\s+chah\w*|mar\s+jaana\s+chah\w*|jeena\s+nahi|jeene\s+ki\s+ichha\s+nahi|aatmahatya|atmahatya|khudkushi|chanipovalani", "suicidal"),
]
_COMPILED = [(re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE), gloss) for pattern, gloss in _GLOSSARY]

# Devanagari and Telugu SCRIPT. Plain substring matching: \b is unusable here
# because the vowel signs inside these words are not "word" characters to the
# re module, so a word boundary would fire in the middle of a word.
_SCRIPT_GLOSSARY = [
    # emergencies (Nov 23) — Devanagari and Telugu script
    (("सीने में दर्द", "छाती में दर्द", "सीने में जकड़न", "ఛాతీ నొప్పి", "ఛాతి నొప్పి", "గుండె నొప్పి", "గుండెల్లో నొప్పి"), "chest pain"),
    (("सांस नहीं", "साँस नहीं", "सांस लेने में तकलीफ", "साँस लेने में तकलीफ", "सांस लेने में दिक्कत", "दम घुट", "ఊపిరి ఆడటం లేదు", "ఊపిరి ఆడడం లేదు"), "trouble breathing"),
    (("बेहोश", "స్పృహ లేదు", "స్పృహ తప్పి"), "unconscious"),
    (("दिल का दौरा", "హార్ట్ అటాక్", "గుండెపోటు"), "having a heart attack"),
    (("मिर्गी", "दौरा पड़"), "seizure"),
    (("खून की उल्टी", "उल्टी में खून", "రక్తం వాంతులు", "రక్తపు వాంతులు"), "vomiting blood"),
    (("खून बंद नहीं", "खून नहीं रुक", "बहुत खून"), "severe bleeding"),
    (("ज्यादा दवा खा", "ज़्यादा दवा खा", "ज्यादा गोलियां खा", "ज़्यादा गोलियाँ खा", "दवा की बोतल खा"), "took too much medicine"),
    (("आत्महत्या", "मरना चाहता", "मरना चाहती", "जीना नहीं"), "suicidal"),
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



# "1 mahine ke bachhe", "do hafte ka baby": a baby's age in Hindi. The under-3-months fever rule
# only understood English ages, so "mere 1 mahine ke bachhe ko bukhar hai" got a product
# recommendation instead of "see a doctor now" (found Nov 23). The age becomes English
# ("1 month old") so every existing age check sees it.
_HINDI_NUMBERS = {"ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "che": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10}
_HINDI_UNITS = {"mahine": "month", "mahina": "month", "maheene": "month", "hafte": "week", "hafta": "week", "haphte": "week", "din": "day", "dino": "day"}
_HINDI_AGE_RE = re.compile(
    r"\b(\d{1,2}|" + "|".join(_HINDI_NUMBERS) + r")\s+(" + "|".join(_HINDI_UNITS) + r")\s+(?:ka|ke|ki)\s+"
    r"(?:bachh?[aei]\w*|bache|bachchi|beta|beti|shishu|baby|nawjaat\w*)\b",
    re.IGNORECASE,
)


def _age_gloss(text: str) -> str | None:
    match = _HINDI_AGE_RE.search(text)
    if not match:
        return None
    raw, unit = match.group(1).lower(), _HINDI_UNITS[match.group(2).lower()]
    number = int(raw) if raw.isdigit() else _HINDI_NUMBERS[raw]
    return f"{number} {unit}{'s' if number != 1 else ''} old"


def gloss(text: str) -> str:
    """The text with English glosses appended for any Hinglish words in it
    (unchanged when there are none, or the English word is already there)."""
    if not text:
        return text
    lower = text.lower()
    added = []
    for pattern, english in _COMPILED:
        if pattern.search(text) and english not in added and not re.search(rf"\b{re.escape(english.lower())}\b", lower):
            added.append(english)
    for words, english in _SCRIPT_GLOSSARY:
        if any(word in text for word in words) and english not in added and not re.search(rf"\b{re.escape(english.lower())}\b", lower):
            added.append(english)
    if not added:
        return text
    return f"{text} ({'; '.join(added)})"
