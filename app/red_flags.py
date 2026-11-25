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

from app import hinglish

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

_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|nor|neither|isn['’]t|aren['’]t|don['’]t|doesn['’]t|didn['’]t|hasn['’]t|haven['’]t)\b|n['’]t\b", re.IGNORECASE)
_NEGATION_LOOKBACK = 25

# Tablets in one go before it counts as an overdose on its own. The highest
# daily total on any product we carry is 8 x 500 mg paracetamol; the message
# rarely says whether it was one sitting, so from 8 up it is escalated.
_OVERDOSE_TABLET_THRESHOLD = 8

_MEDICINE_WORDS = (
    r"(?:tablets?|pills?|capsules?|syrup|bottle|bottles|pack|packs|strips?|doses?|medicine|medication|meds|"
    r"paracetamol|acetaminophen|ibuprofen|cetirizine|pseudoephedrine|guaifenesin|dextromethorphan|antihistamine)"
)
_TAKER = r"(?:i|he|she|they|we|my\s+\w+|(?:the|a)\s+(?:child|kid|baby|toddler)|child|kid|baby|toddler)"
_TOOK = r"(?:took|taken|swallowed|ate|eaten|drank|had|downed|popped)"

_OVERDOSE_RES = [
    re.compile(
        rf"\b{_TAKER}\s+(?:just\s+|accidentally\s+|probably\s+)?(?:have\s+|has\s+|may have\s+|think\s+i\s+)?overdos(?:e|ed|ing)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i|we)\s+(?:think|believe|am afraid)\s+(?:i|we|he|she|they)\s+overdosed\b", re.IGNORECASE),
    re.compile(rf"\b{_TOOK}\s+too\s+(?:many|much)\b", re.IGNORECASE),
    # a CHILD who swallowed a medicine (an adult's "I took 2 paracetamol" is not this; the tablet count handles those)
    re.compile(
        r"\b(?:child|kid|baby|toddler|son|daughter)\s+(?:has\s+|had\s+)?(?:accidentally\s+)?(?:swallowed|ate|eaten|ingested|drank)\b"
        rf".{{0,40}}\b(?:tablets?|pills?|capsules?|medicine|medication|syrup|{_MEDICINE_WORDS})",
        re.IGNORECASE,
    ),
    re.compile(rf"\b{_TOOK}\s+(?:a|the|an|her|his|my|their|its)\s+(?:whole|entire)\s+(?:bottle|pack|strip|box)\b", re.IGNORECASE),
    # Found Nov 20 with the phrasings below: "took a handful of tablets", a parent who "gave my baby
    # double the dose" / "gave my child too much syrup", milligram amounts at once.
    re.compile(rf"\b{_TOOK}\s+(?:a\s+)?(?:handful|fistful|bunch|load)\s+of\b", re.IGNORECASE),
    re.compile(
        r"\b(?:gave|given|giving)\s+(?:my|the|our)\s+(?:\w+\s+){0,3}?(?:baby|child|kid|toddler|son|daughter|infant|(?:year|month|week)s?[- ]old)\b"
        r".{0,60}\b(?:too\s+much|too\s+many|double|twice|triple|(?:\d+|three|four|five)\s+times\s+(?:the|as)|an\s+extra|the\s+wrong|an\s+overdose)\b",
        re.IGNORECASE,
    ),
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
        # Found Nov 20: 25 of 59 everyday emergency phrasings were not recognised ("my chest
        # hurts", "I think I'm having a heart attack", "she collapsed", "turning blue",
        # "stroke", "won't stop bleeding"...). Present-tense, person-in-trouble wording only:
        # "I had a heart attack two years ago" and "what is a stroke" must stay ordinary.
        # chest
        r"\b(?:pain|pressure|tightness|squeezing|crushing)\s+(?:in|on|across)\s+(?:my\s+|the\s+|his\s+|her\s+)?chest\b",
        r"\bchest\s+(?:hurts|is\s+(?:hurting|tight|painful)|feels?\s+(?:tight|heavy|crushed|like))\b",
        r"\b(?:having|getting|suffering)\s+(?:a\s+)?(?:heart\s+attack|stroke)\b|\bthink\w*\b.{0,30}\b(?:heart\s+attack|stroke)\b",
        r"^\s*(?:a\s+)?stroke\s*[?!.]*\s*$",
        # collapse / not responding / not breathing
        r"\b(?:collapsed|collapsing|blacked\s+out|keeled\s+over)\b",
        r"\b(?:not|isn'?t|won'?t|unable\s+to)\s+(?:responding|respond)\b(?!\s+to\s+(?:my\s+|the\s+|his\s+|her\s+)?(?:texts?|messages?|calls?|e-?mails?|dms?|treatment|medicine|medication|antibiotics?))"
        r"|\b(?:not|isn'?t|won'?t|unable\s+to)\s+(?:wake\s+up|waking\s+up|breathing)\b|\bno\s+(?:pulse|response)\b",
        r"\b(?:having|had|got)\s+a\s+fit\b",
        # stroke signs, thunderclap headache, vision
        r"\b(?:weakness|numbness)\s+on\s+one\s+side\b|\bone\s+side\b.{0,25}\b(?:weak|numb|droop\w*)\b",
        r"\bworst\b.{0,20}\bheadache\b|\bheadache\b.{0,40}\bworst\b|\bthunderclap\b|\bdouble\s+vision\b|\bsudden\s+(?:loss\s+of\s+)?vision\b",
        # allergy, bleeding, blood in stool / vomit, severe abdominal pain
        r"\b(?:severe|serious|bad)\s+allergic\b",
        r"\b(?:severe|heavy|heavily|uncontrolled|profuse)\s+bleeding\b|\bbleeding\s+(?:heavily|badly|profusely|a\s+lot)\b"
        r"|\b(?:won'?t|can'?t|not)\s+stop\s+bleeding\b|\bbleeding\s+(?:won'?t|will\s+not)\s+stop\b",
        r"\bpregnan\w*\b.{0,40}\bbleeding\b|\bbleeding\b.{0,40}\bpregnan\w*\b",
        r"\bblood\s+in\s+(?:my\s+|the\s+|his\s+|her\s+)?(?:stool|stools|vomit|urine|poo)\b|\bblack\s+(?:tarry\s+)?(?:stool|stools|poo)\b",
        r"\b(?:severe|unbearable|excruciating)\s+(?:abdominal|stomach|belly|tummy)\s+pain\b",
        # self-harm
        r"\b(?:end(?:ing)?\s+my\s+(?:own\s+)?life|take\s+my\s+own\s+life|kill\s+myself|hurt\s+myself|harm\s+myself|better\s+off\s+dead|"
        r"don'?t\s+want\s+to\s+(?:live|be\s+alive))\b",
        # a baby or child who is limp / not feeding / very sleepy / turning blue
        r"\bturning\s+blue\b",
        # Found Nov 25: 17 of 20 danger signs that go with a fever were not recognised (a rash that
        # does not fade, purple spots, confusion or a hard-to-wake person with a fever, a bulging
        # soft spot, no wet nappies, "can't keep fluids down", drooling with trouble swallowing,
        # light hurting the eyes with a fever / headache, "can't touch chin to chest"). Each is
        # specific enough not to fire on an ordinary message; the product note itself already
        # lists a rash that doesn't fade, confusion and a stiff neck as reasons for emergency care.
        r"\brash\b.{0,40}\b(?:doesn'?t|does\s+not|won'?t|not)\s+(?:fade|go\s+away|blanch|disappear)\b|\bnon-?blanching\b|\bglass\s+test\b",
        r"\bpurple\s+(?:spots|rash|blotches|patches|marks)\b|\bpetechia\w*\b|\bblotchy\s+(?:skin|rash|and\s+cold)\b",
        r"\b(?:fever|feverish|temperature)\b.{0,40}\b(?:confus\w+|drowsy|hard\s+to\s+wake|difficult\s+to\s+wake|can'?t\s+(?:be\s+)?wake\w*|lethargic|floppy|limp)\b"
        r"|\b(?:confus\w+|drowsy|lethargic|floppy|limp)\b.{0,40}\b(?:fever|feverish|temperature)\b",
        r"\b(?:hard|difficult|trouble|struggling)\s+to\s+(?:wake|rouse)\s+(?:him|her|them|my\s+\w+|the\s+\w+|baby|child)\b"
        r"|\b(?:can'?t|cannot|won'?t)\s+(?:be\s+)?(?:woken|wake\s+(?:him|her|them|my\s+\w+|the\s+\w+|baby|child))\b"
        r"|\bwon'?t\s+wake\s+up\b|\bnot\s+waking\s+up\b"
        r"|\b(?:child|baby|toddler|son|daughter|dad|mum|mom|father|mother|patient)\b.{0,40}\b(?:trouble|difficulty|hard|struggling)\s+(?:to\s+)?wak\w+\b",
        r"\b(?:bulging|sunken)\s+(?:soft\s+spot|fontanelle|fontanel)\b",
        r"\bno\s+wet\s+(?:nappies|diapers)\b|\b(?:not|hasn'?t|haven'?t)\s+(?:peed|urinated|passed\s+urine|had\s+a\s+wee)\b.{0,30}\b(?:all\s+day|for\s+(?:\d+|a|an)\s+(?:hours?|day)|since\s+yesterday|in\s+\d+\s+hours)\b"
        r"|\bnot\s+(?:peeing|urinating)\s+(?:at\s+all|much)\b|\bno\s+urine\b|\bcan'?t\s+keep\s+(?:any\s+)?(?:fluids?|water|liquids?)\s+down\b",
        r"\b(?:fever|feverish|temperature|baby|infant|newborn)\b.{0,40}\b(?:not\s+drinking|won'?t\s+drink|refus\w+\s+to\s+drink|refus\w+\s+(?:all\s+)?(?:fluids|drinks))\b",
        r"\bdrool\w*\b.{0,40}\b(?:swallow\w*|breath\w*)\b|\b(?:difficulty|trouble|can'?t)\s+swallow\w*\b.{0,40}\bdrool\w*",
        r"\b(?:fever|headache)\b.{0,50}\b(?:light\s+(?:hurts|hurting)|sensitive\s+to\s+(?:the\s+)?light|photophobia)\b|\b(?:light\s+hurts|photophobia)\b",
        # Found Nov 25 (reactions to a medicine, and signs that need urgent care): yellow eyes or skin,
        # swelling of the face / lips / tongue, hives with a throat or breathing symptom, wheezing,
        # "can't catch my breath", about to faint, can't pass urine, slurred or lost speech, a
        # numb or drooping half of the face, arm or jaw pain with sweating or nausea.
        r"\bjaundice\b|\byellow(?:ing|ed)?\s+(?:eyes|skin)\b|\b(?:eyes|skin)\s+(?:have\s+|has\s+|are\s+|is\s+)?(?:turned\s+|turning\s+|gone\s+)?yellow\b",
        r"\bswelling\s+(?:of|in|around)\s+(?:the\s+|my\s+|his\s+|her\s+)?(?:face|lips|tongue|throat|mouth|eyes)\b",
        r"\bhives\b.{0,40}\b(?:throat|breath\w*|swelling|swollen|face|lips|tongue)\b|\b(?:throat|breath\w*|swelling|swollen|face|lips|tongue)\b.{0,40}\bhives\b",
        r"\bwheez(?:e|es|ing)\b|\b(?:can'?t|cannot|unable\s+to)\s+catch\s+(?:my|his|her|their)\s+breath\b|\bgasping\b",
        r"\b(?:about\s+to|going\s+to|feel\s+like\s+(?:i'?m\s+|i\s+am\s+)?going\s+to)\s+(?:faint|pass\s+out|black\s+out|collapse)\b|\b(?:feel(?:ing)?|very|really)\s+faint\b",
        r"\b(?:can'?t|cannot|unable\s+to|haven'?t\s+been\s+able\s+to)\s+(?:pass\s+urine|pee|urinate|wee)\b(?!\s+(?:in\s+public|in\s+front|at\s+work|at\s+school))",
        r"\bspeech\s+(?:is\s+|has\s+become\s+|got\s+)?slurred\b|\b(?:trouble|difficulty)\s+speaking\b|\b(?:can'?t|cannot|unable\s+to)\s+speak\b",
        r"\b(?:half|one\s+side)\s+(?:of\s+)?(?:my|his|her|the)?\s*(?:face|body)\b.{0,15}\b(?:numb|weak|droop\w*|paralys\w*)\b|\bnumb(?:ness)?\s+(?:in|on)\s+(?:my\s+|his\s+|her\s+|the\s+)?(?:face|arm)\b",
        r"\b(?:pain|ache)\s+(?:in|down|radiating\s+to|spreading\s+to)\s+(?:my\s+|his\s+|her\s+|the\s+)?(?:left\s+)?(?:arm|jaw)\b.{0,40}\b(?:sweat\w*|nausea|breath\w*|chest)\b"
        r"|\b(?:sweat\w*|nausea)\b.{0,40}\b(?:arm|jaw)\s+pain\b|\bjaw\s+pain\b.{0,30}\bsweat",
        r"\b(?:palpitations?|heart\s+(?:is\s+)?(?:racing|pounding|fluttering)|racing\s+heart|irregular\s+heartbeat|fast\s+heartbeat)\b.{0,50}\b(?:faint\w*|dizz\w+|chest|short\s+of\s+breath|sweat\w*|pain)\b",
        r"\b(?:can'?t|cannot|unable\s+to|won'?t)\s+(?:touch|put|bring|bend)\b.{0,25}\bchin\b.{0,12}\bchest\b|\bchin\s+to\s+(?:his\s+|her\s+|my\s+|the\s+)?chest\b|\bneck\s+(?:is\s+)?stiff\b",
        r"\b(?:baby|infant|newborn|toddler|child)\b.{0,40}\b(?:not\s+feeding|won'?t\s+feed|refus\w*\s+(?:feeds?|to\s+feed)|very\s+sleepy|too\s+sleepy|limp|floppy)\b",
    )
]

_FEVER_RE = re.compile(
    r"\b(?:fever|feverish|temperature|temp|hot\s+to\s+the\s+touch|burning\s+up|warm\s+to\s+the\s+touch)\b|"
    r"\b(?:feels?|feeling|running)\s+(?:a\s+)?(?:very\s+|really\s+|so\s+)?(?:warm|hot)\b",
    re.IGNORECASE,
)
_BABY_WORD_RE = re.compile(r"\b(?:baby|babies|infant|newborn|bub|little\s+one|son|daughter)\b", re.IGNORECASE)
_UNDER_AGE_RE = re.compile(
    r"\b(?:under|less\s+than|younger\s+than|below)\s+(?:the\s+age\s+of\s+)?(\d|one|two|three)\s*months?\b", re.IGNORECASE
)
_DURATION_AFTER_RE = re.compile(r"\s*(?:now|already|so\s+far|ago|into|long|straight|in\s+a\s+row)\b", re.IGNORECASE)
_DURATION_BEFORE_RE = re.compile(r"\b(?:for|since|past|last|over|about|almost|nearly)\s+(?:the\s+)?(?:last\s+|past\s+)?$", re.IGNORECASE)
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


_CURLY = str.maketrans({"\u2018": "'", "\u2019": "'"})


_COUNT_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "dozen": 12,
}
_COUNT_WORD_RE = re.compile(r"\b(?:a\s+)?(" + "|".join(_COUNT_WORDS) + r")\b", re.IGNORECASE)
_AT_ONCE = r"(?:at\s+once|all\s+at\s+once|in\s+one\s+(?:go|sitting|hit)|in\s+one\s+dose)"
# Several tablets in a single dose is already well past the label (one dose is one or two
# tablets), without reaching the 8-in-a-day threshold.
_AT_ONCE_THRESHOLD = 4
_AT_ONCE_COUNT_RE = re.compile(
    rf"\b{_TOOK}\s+(\d{{1,3}})\s+(?:\w+\s+){{0,3}}{_MEDICINE_WORDS}\b.{{0,40}}\b{_AT_ONCE}\b", re.IGNORECASE
)
_MG_RE = re.compile(rf"\b{_TOOK}\b.{{0,30}}?\b(\d{{4,5}})\s*mg\b(?P<tail>.{{0,30}})", re.IGNORECASE)
_MG_DAILY_MAX = 4000      # the most any product here allows in 24 hours
_MG_AT_ONCE = 3000


def is_overdose(text: str) -> bool:
    text = text.translate(_CURLY)
    if any(_has_unnegated_match(p, text) for p in _OVERDOSE_RES):
        return True
    # "took ten paracetamol" counts like "took 10 paracetamol"
    counted = _COUNT_WORD_RE.sub(lambda m: str(_COUNT_WORDS[m.group(1).lower()]), text)
    for match in _TABLET_COUNT_RE.finditer(counted):
        before = counted[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
        if int(match.group(1)) >= _OVERDOSE_TABLET_THRESHOLD and not _NEGATION_RE.search(before):
            return True
    for match in _AT_ONCE_COUNT_RE.finditer(counted):
        before = counted[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
        if int(match.group(1)) >= _AT_ONCE_THRESHOLD and not _NEGATION_RE.search(before):
            return True
    for match in _MG_RE.finditer(text):
        before = text[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
        amount = int(match.group(1))
        at_once = bool(re.search(_AT_ONCE, match.group("tail"), re.IGNORECASE))
        if (amount >= _MG_DAILY_MAX or (at_once and amount >= _MG_AT_ONCE)) and not _NEGATION_RE.search(before):
            return True
    return False


_EXTREME_C = 41.0      # 105.8 F: hyperpyrexia, not a "see a doctor soon"
_EXTREME_F = 105.8


def _extreme_temperature(text: str) -> bool:
    """A temperature of 41 C / 105.8 F or more. Same unit rules as _high_temperature."""
    for match in _TEMP_RE.finditer(text):
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        if unit.startswith("c"):
            if _EXTREME_C <= value <= 43:
                return True
        elif unit.startswith("f"):
            if _EXTREME_F <= value <= 110:
                return True
        elif "." in match.group(1):
            if _EXTREME_C <= value <= 43 or _EXTREME_F <= value <= 110:
                return True   # no unit: nobody has a 41.5 F fever or a 105.9 C one, so the number says which
        elif _EXTREME_F <= value <= 110:
            return True
        elif 41 <= value <= 43:
            # a whole 41-43 is also an age or a dose ("I am 41", "41 mg"): only right after a fever word
            number = int(value)
            if re.search(rf"\b(?:fever|temperature|temp)\s*(?:of|is|was|at|:)?\s*{number}\b(?!\s*(?:years?|yo|y/o|mg|ml|kg|%|days?|weeks?|months?))", text, re.IGNORECASE):
                return True
    return False


def is_emergency(text: str) -> bool:
    text = text.translate(_CURLY)  # "can’t breathe" typed on a phone
    if _FEVER_RE.search(text) and _extreme_temperature(text):
        return True
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
            # ...or a baby is named and the number is not a duration: "my infant has a fever, she is
            # 7 weeks", "my baby (6 weeks) has a high temperature" (found Nov 25).
            nearby = text[max(0, match.start() - 45): match.end() + 45]
            if (
                _BABY_WORD_RE.search(nearby)
                and (unit.startswith("day") or number <= 12)
                and not _DURATION_BEFORE_RE.search(text[: match.start()])
                and not _DURATION_AFTER_RE.match(text[match.end():])
            ):
                return True
        elif unit.startswith("month") and number < 3:
            return True
    # "baby under 3 months with fever": the age is a ceiling, and 3 months is still under it
    under = _UNDER_AGE_RE.search(text)
    if under and _BABY_WORD_RE.search(text):
        raw = under.group(1).lower()
        if (int(raw) if raw.isdigit() else {"one": 1, "two": 2, "three": 3}[raw]) <= 3:
            return True
    return False


def check(text: str) -> str | None:
    """The fixed reply for a message describing an emergency, or None."""
    if not text:
        return None
    # Hindi / Telugu (romanised or in script): glossed into the English words below. chat()
    # glosses ordinary messages already; this also covers the moments it does not (an order
    # is waiting for its address), when an emergency must still be seen.
    text = hinglish.gloss(text.translate(_CURLY))
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


# Durations with no number ("feverish for the past week", "for a few days", "all week", "for
# days now", "since last week"): found Nov 24, none of these got the see-a-doctor note, while
# "for 3 days" did. "a couple of days" is two days and is deliberately not here.
_VAGUE_LONG_RE = re.compile(
    r"\b(?:for|over|during)\s+(?:the\s+)?(?:past|last)\s+(?:week|few\s+days|several\s+days|\d+\s+days)\b|"
    r"\b(?:a\s+few|few|several|many)\s+days\b(?!\s+ago\b)|\ball\s+week\b|\bthis\s+whole\s+week\b|\bfor\s+(?:days|weeks|ages)\b|"
    r"\bsince\s+last\s+(?:week|month)\b|\bfor\s+over\s+a\s+week\b",
    re.IGNORECASE,
)


def _long_fever(text: str) -> bool:
    if _VAGUE_LONG_RE.search(text):
        return True
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
