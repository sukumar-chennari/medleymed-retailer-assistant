"""Brand and medicine names the shop does not stock. Found Oct 22: "do you
have Advil", "I want Tylenol", "I need Crocin", "do you have Dolo 650" were all
answered "This demo only handles fever and cold OTC guidance — I can't help
with that here", which is wrong twice over: they are fever medicines, and each
of those brands is just the active ingredient we DO stock. The honest answer is
"we don't carry that brand, but here is the same ingredient". Brands whose
active ingredient is not a product we carry (NyQuil, Benadryl, aspirin,
antibiotics, vitamins) get "we don't carry that" and a pointer to what we do.
Dependency-free on purpose; the equivalence is the brand's own active
ingredient, nothing clinical is inferred."""

import re

# brand / alias -> active ingredient as it appears in catalog.json
BRAND_INGREDIENT = {
    "tylenol": "Paracetamol", "panadol": "Paracetamol", "crocin": "Paracetamol", "dolo": "Paracetamol",
    "calpol": "Paracetamol", "acetaminophen": "Paracetamol",
    "advil": "Ibuprofen", "motrin": "Ibuprofen", "brufen": "Ibuprofen", "nurofen": "Ibuprofen",
    "zyrtec": "Cetirizine", "cetzine": "Cetirizine",
    "sudafed": "Pseudoephedrine",
    "mucinex": "Guaifenesin",
    "delsym": "Dextromethorphan",
}
# Aliases that are the ingredient itself, not a brand.
_SAME_AS_INGREDIENT = {"acetaminophen"}

NOT_STOCKED = {
    "aspirin", "nyquil", "dayquil", "benadryl", "robitussin", "vicks", "otrivin", "strepsils", "halls",
    "lozenges", "amoxicillin", "azithromycin", "antibiotic", "antibiotics", "augmentin", "vitamin",
    "vitamins", "omeprazole", "naproxen", "diclofenac", "combiflam", "disprin", "dayquil", "theraflu",
    "tussin", "codeine", "ors", "zinc",
}

_AVAILABILITY_RE = re.compile(
    r"\b(?:do you (?:have|sell|stock|carry|offer)|have you got|got any|any|"
    r"i (?:want|need|would like|'d like)|i'd like|can i (?:get|buy|have)|looking for|"
    r"(?:give|get|sell) me|is .{1,30} available|do you do)\b",
    re.IGNORECASE,
)
_NOT_AVAILABILITY_RE = re.compile(
    r"\b(?:better|vs|versus|difference|safe|side\s*effects?|dos(?:e|age)|how (?:much|many|long)|"
    r"interact\w*|work|works|ingredients?|pregnan\w*|alcohol|drowsy|take|taking|took)\b",
    re.IGNORECASE,
)
_MAX_WORDS = 12


def _mentions(text: str, names) -> list[str]:
    lower = text.lower()
    return [n for n in names if re.search(rf"\b{re.escape(n)}\b", lower)]


def classify(text: str) -> tuple[list[str], list[str]] | None:
    """(brands with a stocked ingredient, other unstocked names) when the text
    asks for a medicine we don't carry, else None."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > _MAX_WORDS or "@" in text:
        return None
    if not _AVAILABILITY_RE.search(text) or _NOT_AVAILABILITY_RE.search(text):
        return None
    brands = _mentions(text, BRAND_INGREDIENT)
    others = _mentions(text, NOT_STOCKED)
    if not brands and not others:
        return None
    return brands, others


def is_ingredient_itself(brand: str) -> bool:
    return brand in _SAME_AS_INGREDIENT


# A reply that RECOMMENDS a medicine we don't carry. Found Oct 22-26: the model's
# free text said "I suggest taking Acetaminophen (Tylenol)", "ibuprofen (Advil,
# Motrin)", "you might consider taking ... Benadryl" — products that are not in
# the catalog, so they cannot be ordered and were never vetted against the
# knowledge base. Mentioning a brand informationally ("Tylenol is a brand of
# paracetamol") is fine; recommending one is not, unless the customer named it.
_RECOMMEND_VERBS = (
    r"(?:recommend(?:ed|ing)?|suggest(?:ed|ing)?|consider(?:ing)?|try(?:ing)?|take|taking|use|using|opt\s+for|go\s+with|"
    r"best\s+(?:option|choice)\s+is|you\s+(?:can|could|might|may|should)\s+(?:take|try|use))"
)
_RECOMMENDABLE_OFF_CATALOG = sorted(
    (n for n in {*BRAND_INGREDIENT, *NOT_STOCKED} if n not in {"acetaminophen", "ors"}), key=len, reverse=True
)


def recommended_off_catalog(reply: str, user_text: str = "") -> list[str]:
    """Brand/medicine names the reply recommends that we don't stock and that
    the customer did not name themselves."""
    found = []
    user_lower = user_text.lower()
    for name in _RECOMMENDABLE_OFF_CATALOG:
        if re.search(rf"\b{re.escape(name)}\b", user_lower):
            continue
        if re.search(rf"\b{_RECOMMEND_VERBS}\b[^.\n]{{0,80}}\b{re.escape(name)}\b", reply, re.IGNORECASE):
            found.append(name)
    return found
