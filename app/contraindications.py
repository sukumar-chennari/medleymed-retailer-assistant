"""Conditions the customer states about themselves, checked against what each
product's own information says. Found Oct 21 by telling the live bot about
a condition and asking for a recommendation:
  - "I have a fever and stomach ulcer" and "I'm on warfarin and have a
    headache and fever" were both offered Ibuprofen 200mg Tablets, whose
    product information says it is not recommended for people with a history
    of stomach ulcers;
  - "I have high blood pressure and a stuffy nose" was recommended
    Pseudoephedrine, whose information says it is not recommended for high
    blood pressure ("However, please note..." was the model's own addition);
  - "I'm allergic to ibuprofen" did not stop the deterministic lists from
    containing it.
Every rule below comes from the knowledge base's own Warnings sections (cited
in the reason), except two that are plain logic rather than medical claims: a
stated allergy to an ingredient excludes products containing it, and a
condition the knowledge base is silent on (blood thinners) is reported as "not
covered — ask a pharmacist" rather than judged. Dependency-free on purpose.
"""

import contextvars
import re

_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|nor|neither|isn't|aren't|don't|doesn't|didn't|hasn't|haven't|free\s+of|history\s+of\s+no)\b|n't\b",
    re.IGNORECASE,
)
_NEGATION_LOOKBACK = 25

# condition -> (label shown to the customer, pattern)
_CONDITIONS = {
    "ulcer": ("a stomach ulcer", r"\b(?:stomach|peptic|gastric|duodenal)\s+ulcers?\b|\bulcers?\b"),
    "kidney": ("kidney problems", r"\bkidney\s+(?:problems?|disease|failure|issues?|stones?|damage)\b|\brenal\b|\bdialysis\b|\bckd\b"),
    "heart": ("a heart condition", r"\bheart\s+(?:disease|condition|conditions|problems?|failure|attack|issues?)\b|\bcardiac\b|\bangina\b|\barrhythmia\b|\bpacemaker\b"),
    "high_bp": ("high blood pressure", r"\bhigh\s+blood\s+pressure\b|\bhypertensi(?:on|ve)\b|\bhigh\s+bp\b|\bbp\s+(?:is\s+)?high\b"),
    "thyroid": ("a thyroid condition", r"\bthyroid\b|\bhyperthyroid\w*|\bhypothyroid\w*|\bgraves\b"),
    "pregnant": ("pregnancy", r"\bpregnan\w*|\bexpecting\b|\btrimester\b"),
    "asthma": ("asthma", r"\basthma\w*|\bemphysema\b|\bcopd\b"),
    "smoker": ("a smoker's cough", r"\bsmoker\w*|\bsmoking\b|\bi\s+smoke\b"),
    "liver": ("liver problems", r"\bliver\s+(?:problems?|disease|damage|issues?)\b|\bcirrhosis\b|\bhepatitis\b"),
    "alcohol": ("regular alcohol use", r"\bdrinks?\s+(?:alcohol\s+)?(?:daily|regularly|heavily|a\s+lot)\b|\bheavy\s+drinker\b|\balcoholic\b|\bdrink\s+(?:alcohol\s+)?every\s+day\b"),
    "maoi": ("an MAOI antidepressant", r"\bmaoi\b|\bphenelzine\b|\bselegiline\b|\btranylcypromine\b|\bmoclobemide\b"),
    "antidepressant": ("an antidepressant", r"\bantidepressants?\b"),
    "nsaid_use": ("another anti-inflammatory", r"\b(?:taking|on|using|took|take)\s+(?:an?\s+)?(?:aspirin|naproxen|diclofenac|nsaids?|ibuprofen|brufen|advil)\b"),
    "paracetamol_use": ("other paracetamol", r"\b(?:taking|on|using|took|take)\s+(?:some\s+|other\s+|more\s+)?(?:paracetamol|acetaminophen|crocin|dolo|panadol|tylenol|calpol)\b"),
    "blood_thinner": ("a blood thinner", r"\bwarfarin\b|\bblood\s+thinners?\b|\banticoagulants?\b|\bclopidogrel\b|\bapixaban\b|\brivaroxaban\b|\bheparin\b"),
}
_COMPILED = {key: (label, re.compile(pattern, re.IGNORECASE)) for key, (label, pattern) in _CONDITIONS.items()}

_NOT_RECOMMENDED = "our product information says it isn't recommended in that case without a doctor's advice"
_CONSULT = "our product information says to consult a doctor or pharmacist first in that case"
_NOT_COVERED = "our product information doesn't cover that, so please check with a pharmacist before taking it"

# (condition, product id, "exclude" | "note", reason template)
_RULES = [
    ("ulcer", "fev-003", "exclude", _NOT_RECOMMENDED),
    ("kidney", "fev-003", "exclude", _NOT_RECOMMENDED),
    ("heart", "fev-003", "exclude", _NOT_RECOMMENDED),
    ("pregnant", "fev-003", "exclude", "our product information says to avoid it during pregnancy unless a doctor advises otherwise"),
    ("nsaid_use", "fev-003", "exclude", "our product information says not to combine it with other anti-inflammatories"),
    ("blood_thinner", "fev-003", "exclude", _NOT_COVERED),
    ("asthma", "fev-003", "note", _NOT_COVERED),
    ("heart", "col-002", "exclude", _NOT_RECOMMENDED),
    ("high_bp", "col-002", "exclude", _NOT_RECOMMENDED),
    ("thyroid", "col-002", "exclude", _NOT_RECOMMENDED),
    ("heart", "col-003", "exclude", _NOT_RECOMMENDED),
    ("high_bp", "col-003", "exclude", _NOT_RECOMMENDED),
    ("paracetamol_use", "col-003", "exclude", "it already contains paracetamol, and our product information warns against doubling up"),
    ("kidney", "col-001", "note", _CONSULT),
    ("asthma", "col-005", "exclude", _NOT_RECOMMENDED),
    ("smoker", "col-005", "exclude", _NOT_RECOMMENDED),
    ("maoi", "col-004", "exclude", "our product information says it must not be combined with MAOI antidepressants"),
    ("antidepressant", "col-004", "note", "our product information says not to combine it with certain antidepressants (MAOIs) — check yours with a pharmacist"),
    ("liver", "fev-001", "note", _CONSULT),
    ("liver", "fev-002", "note", _CONSULT),
    ("alcohol", "fev-001", "note", _CONSULT),
    ("alcohol", "fev-002", "note", _CONSULT),
    ("paracetamol_use", "fev-001", "exclude", "our product information says not to combine it with other paracetamol products"),
    ("paracetamol_use", "fev-002", "exclude", "our product information says not to combine it with other paracetamol products"),
    ("paracetamol_use", "fev-004", "exclude", "our product information says not to combine it with other paracetamol products"),
]

# Ingredient / brand words -> product ids containing that ingredient.
_PARACETAMOL = ("fev-001", "fev-002", "fev-004", "col-003")
_ALLERGY_TERMS = [
    (r"ibuprofen|nsaids?|aspirin|brufen|advil|motrin|anti-?inflammator\w*", ("fev-003",)),
    (r"paracetamol|acetaminophen|tylenol|panadol|crocin|dolo|calpol", _PARACETAMOL),
    (r"cetirizine|zyrtec", ("col-001",)),
    (r"antihistamines?", ("col-001", "col-003", "col-006")),
    (r"pseudoephedrine|sudafed", ("col-002",)),
    (r"decongestants?", ("col-002", "col-003")),
    (r"phenylephrine", ("col-003",)),
    (r"chlorpheniramine", ("col-003", "col-006")),
    (r"guaifenesin", ("col-005",)),
    (r"dextromethorphan", ("col-004", "col-006")),
]
_ALLERGY_RES = [(re.compile(p, re.IGNORECASE), ids) for p, ids in _ALLERGY_TERMS]
_ALLERGY_PHRASE_RE = re.compile(r"\ballerg(?:ic|y|ies)\b[^.!?;]{0,60}", re.IGNORECASE)
_ALLERGY_BEFORE_RE = re.compile(r"([A-Za-z\-/ ]{3,40}?)\s+allergy\b", re.IGNORECASE)


def _unnegated(pattern: re.Pattern, text: str) -> bool:
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
        if not _NEGATION_RE.search(before):
            return True
    return False


def conditions_in(text: str) -> set[str]:
    """The condition keys the text states (negated mentions are ignored)."""
    return {key for key, (_, pattern) in _COMPILED.items() if _unnegated(pattern, text)}


def allergic_product_ids(text: str) -> dict[str, str]:
    """product id -> the allergy phrase that excludes it. A negated allergy
    ("I'm not allergic to ibuprofen") excludes nothing."""
    found: dict[str, str] = {}
    matches = list(_ALLERGY_PHRASE_RE.finditer(text)) + list(_ALLERGY_BEFORE_RE.finditer(text))
    for match in matches:
        before = text[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
        keyword_at = match.group(0).lower().find("allerg")
        inside_before = match.group(0)[: max(keyword_at, 0)]
        if _NEGATION_RE.search(before) or _NEGATION_RE.search(inside_before):
            continue
        for pattern, ids in _ALLERGY_RES:
            hit = pattern.search(match.group(0))
            if hit:
                for product_id in ids:
                    found.setdefault(product_id, hit.group(0).lower())
    return found


def label(condition: str) -> str:
    return _COMPILED[condition][0]


def concerns(text: str) -> dict[str, list[tuple[str, str]]]:
    """product id -> [(mode, reason sentence), ...] for everything the text
    states about the customer. Mode is "exclude" or "note"."""
    result: dict[str, list[tuple[str, str]]] = {}
    for condition in conditions_in(text):
        for rule_condition, product_id, mode, template in _RULES:
            if rule_condition == condition:
                reason = template.format(c=label(condition)) if "{c}" in template else template
                result.setdefault(product_id, []).append((mode, f"you mentioned {label(condition)}, and {reason}"))
    for product_id, phrase in allergic_product_ids(text).items():
        result.setdefault(product_id, []).append(
            ("exclude", f"you mentioned an allergy to {phrase}, which it contains")
        )
    return result


# The text of the message being handled right now. start_order runs while a
# turn is in progress, before that turn's messages are saved to the session
# history, so "I have an ulcer, order Ibuprofen 200mg Tablets" would otherwise
# not be seen by it.
_current_text: contextvars.ContextVar[str] = contextvars.ContextVar("current_text", default="")


def set_current_text(text: str) -> None:
    _current_text.set(text or "")


def stated_text(history_user_texts: list[str]) -> str:
    return "\n".join([*history_user_texts, _current_text.get()])


_RECOMMENDING_RE = re.compile(
    r"\((?:fev|col)-\d{3}\)|would be a good fit|would you like to (?:place (?:an )?)?order|which one would you like|\bi(?:'d| would)?\s+recommend",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[-*]\s+.*\((?:fev|col)-\d{3}\)")


def refusal_for(product: dict, user_text: str) -> str | None:
    """Why this product shouldn't be ordered for this customer, or None."""
    reasons = [r for mode, r in concerns(user_text).get(product["id"], []) if mode == "exclude"]
    if not reasons:
        return None
    return (
        f"I can't set up an order for {product['name']}: {reasons[0]}. "
        f"Please check with a doctor or pharmacist first — I'm happy to look at other options."
    )


def screen_reply(reply: str, user_text: str, catalog: list[dict]) -> tuple[str, set[str]]:
    """Removes from a recommendation the products this customer's stated
    conditions rule out, and adds a note for the ones that need care. Works on
    the finished reply, so it covers the deterministic lists and the model's
    own prose alike. Returns (reply, ids that were excluded)."""
    conc = concerns(user_text)
    if not conc or not _RECOMMENDING_RE.search(reply):
        return reply, set()

    lower = reply.lower()
    named = [p for p in catalog if re.search(rf"\b{re.escape(p['id'])}\b", lower) or p["name"].lower() in lower]
    excluded = {p["id"]: p for p in named if any(m == "exclude" for m, _ in conc.get(p["id"], []))}
    noted = [p for p in named if p["id"] not in excluded and any(m == "note" for m, _ in conc.get(p["id"], []))]
    if not excluded and not noted:
        return reply, set()

    def reason_of(product_id: str, mode: str) -> str:
        return next(r for m, r in conc[product_id] if m == mode)

    lines = reply.split("\n")
    bullets = [i for i, line in enumerate(lines) if _BULLET_RE.match(line)]
    kept_lines = [
        line for i, line in enumerate(lines)
        if not (i in bullets and any(re.search(rf"\({re.escape(pid)}\)", line) for pid in excluded))
    ]
    removed_any = len(kept_lines) != len(lines)
    remaining_bullets = [line for line in kept_lines if _BULLET_RE.match(line)]
    single_recommendation = not bullets and any(
        lower.startswith(p["name"].lower()) for p in excluded.values()
    )

    if excluded and ((bullets and not remaining_bullets) or single_recommendation):
        first = next(iter(excluded.values()))
        text = (
            f"I can't recommend {first['name']} for you: {reason_of(first['id'], 'exclude')}. "
            "Please check with a doctor or pharmacist first — I'm happy to look at other options."
        )
        return text, set(excluded)

    out = "\n".join(kept_lines) if removed_any else reply
    notices = [
        f"- {p['name']}: {reason_of(p['id'], 'exclude')}"
        for p in excluded.values()
    ]
    if notices:
        heading = "Left out because of what you told me:" if removed_any else "Please be careful with:"
        out = f"{out.rstrip()}\n\n{heading}\n" + "\n".join(notices)
        if not removed_any:
            out += "\nI'd suggest not choosing it without checking with a doctor or pharmacist."
    for p in noted:
        out += f"\n\nNote on {p['name']}: {reason_of(p['id'], 'note')}."
    return out, set(excluded)
