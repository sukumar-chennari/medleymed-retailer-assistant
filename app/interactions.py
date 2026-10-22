"""Questions of the form "can I take X and Y together?". Found Oct 22: the
model answers these from general knowledge — "cetirizine and pseudoephedrine
can be taken together", "yes, you can take ibuprofen and paracetamol together"
— while the knowledge base has specific, different statements: never double up
on paracetamol (it names Cold & Flu Multi-Symptom Relief explicitly), never
combine two decongestants, never combine two dextromethorphan products, and
the children's cold & cough syrup must not go with any other antihistamine or
cough suppressant. Those warnings are applied in code; a combination the
knowledge base says nothing about gets "our product information doesn't cover
this, ask a pharmacist" instead of an improvised yes. Dependency-free: the
catalog is passed in."""

import re

# active ingredient word -> the classes it belongs to
_CLASSES = {
    "paracetamol": {"paracetamol"},
    "ibuprofen": {"nsaid"},
    "cetirizine": {"antihistamine"},
    "chlorpheniramine": {"antihistamine"},
    "pseudoephedrine": {"decongestant"},
    "phenylephrine": {"decongestant"},
    "dextromethorphan": {"cough_suppressant", "dextromethorphan"},
    "guaifenesin": {"expectorant"},
}
_CLASS_LABEL = {
    "paracetamol": "paracetamol",
    "decongestant": "a decongestant",
    "dextromethorphan": "dextromethorphan",
    "antihistamine": "an antihistamine",
    "cough_suppressant": "a cough suppressant",
}

# Each warning is the knowledge base's own wording for that product.
_WARNINGS = {
    "paracetamol": "Do not combine with other products that also contain paracetamol/acetaminophen, as this can lead to accidental overdose.",
    "decongestant": "Avoid combining with other decongestants or stimulants.",
    "dextromethorphan": "Do not combine with other cough/cold products that also contain dextromethorphan.",
    "children_syrup": "Do not use alongside any other product containing an antihistamine or cough suppressant.",
}

_TOGETHER_RE = re.compile(
    r"\b(?:together|combine[ds]?|combining|at\s+the\s+same\s+time|same\s+time|along\s*with|alongside|mix(?:ing)?|"
    r"both|with\s+each\s+other|and\s+also|in\s+combination)\b|\b(?:and|with|plus|or)\b",
    re.IGNORECASE,
)
_QUESTION_STARTS = {"can", "could", "should", "may", "is", "are", "will", "do", "does", "would", "what"}
_NEEDS_PAIRING_RE = re.compile(r"\b(?:together|combine[ds]?|combining|same\s+time|along\s*with|alongside|mix(?:ing)?|both|with\s+each\s+other|in\s+combination)\b|\b(?:take|taking|use|using|give|giving|have)\b.{0,50}\b(?:and|with|plus)\b", re.IGNORECASE)
_NOT_THIS_RE = re.compile(r"\b(?:order|buy|price|cost|cheapest|dos(?:e|age)|side\s*effects?|how\s+much|how\s+many)\b", re.IGNORECASE)


def _product_classes(product: dict) -> set[str]:
    classes: set[str] = set()
    for word in re.findall(r"[a-z]+", product["active_ingredient"].lower()):
        classes |= _CLASSES.get(word, set())
    return classes


def mentions(text: str, catalog: list[dict]) -> list[tuple[str, set[str], str | None]]:
    """(label, classes, product id or None) for each distinct medicine the
    text names — by full name or id (one product) or by active ingredient."""
    lower = " ".join(text.lower().replace("’", "'").split())
    found = []
    for product in catalog:
        name = product["name"].lower()
        if name in lower:
            found.append((product["name"], _product_classes(product), product["id"]))
            lower = lower.replace(name, " ")
    for product in catalog:
        if product["id"] in lower:
            found.append((product["name"], _product_classes(product), product["id"]))
            lower = lower.replace(product["id"], " ")
    tokens = set(re.findall(r"[a-z]+", lower))
    for word, classes in _CLASSES.items():
        if word in tokens:
            found.append((word.title(), set(classes), None))
    # one entry per distinct label
    seen, unique = set(), []
    for item in found:
        if item[0] not in seen:
            seen.add(item[0])
            unique.append(item)
    return unique


def is_combination_question(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > 24 or "@" in text or _NOT_THIS_RE.search(text):
        return False
    framed = words[0] in _QUESTION_STARTS or "?" in text
    return framed and bool(_NEEDS_PAIRING_RE.search(text))


def combination_reply(text: str, catalog: list[dict]) -> str | None:
    if not is_combination_question(text):
        return None
    found = mentions(text, catalog)
    if len(found) < 2:
        return None
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            (a_label, a_classes, a_id), (b_label, b_classes, b_id) = found[i], found[j]
            shared = [c for c in ("paracetamol", "decongestant", "dextromethorphan") if c in a_classes and c in b_classes]
            if shared:
                kind = shared[0]
                return (
                    f"No — {a_label} and {b_label} both contain {_CLASS_LABEL[kind]}, and our product information "
                    f"warns: \"{_WARNINGS[kind]}\" Please don't take them together; a pharmacist can help if you're unsure."
                )
            children = [(x_label, x_id, y_classes) for (x_label, _, x_id), (_, y_classes, _) in
                        ((found[i], found[j]), (found[j], found[i])) if x_id == "col-006"]
            for x_label, _, other_classes in children:
                other = b_label if x_label == a_label else a_label
                if other_classes & {"antihistamine", "cough_suppressant"}:
                    return (
                        f"No — for {x_label}, our product information warns: \"{_WARNINGS['children_syrup']}\" "
                        f"{other} is one. Please don't use them together; a pharmacist can help if you're unsure."
                    )
    labels = " and ".join(label for label, _, _ in found[:2])
    return (
        f"Our product information doesn't say whether {labels} can be taken together, so I can't confirm it. "
        "Please ask a pharmacist or doctor before combining them."
    )
