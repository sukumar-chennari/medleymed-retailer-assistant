"""Messages that describe a cough TOGETHER with a fever and/or other cold
symptoms ("I have a fever, cough and runny nose"). Found Oct 28:
- "I'm an adult with a fever and a dry cough" and "... a wet cough, runny nose
  and fever" were answered with the cough product ONLY — the fever and the
  runny nose were silently dropped (the "completeness" limitation documented
  since the clarification code was written);
- "I have a fever, cough, and runny nose" asked "dry or chesty?", and the
  natural reply "for myself" was then declined as out of scope.
This module asks ONE combined question for whatever is missing (age, cough
type), keeps the partial answers in the pending-clarification string, and
recommends per symptom. The products are the ones the existing clarifying
rules already choose; the only new decision is that Cold & Flu Multi-Symptom
Relief is left out when fever products are offered too, because it already
contains paracetamol and the knowledge base warns against doubling up.

Pending state is the string "multi:<groups>;age=<child|adult|>;cough=<dry|wet|>".
Only messages that mention a cough plus at least one other group come here;
everything else keeps the existing logic."""

import re

from app import store, tools

PREFIX = "multi:"

_COLD_WORDS_RE = re.compile(
    r"\b(?:cold|flu|runny|nose|nasal|sneez\w*|congest\w*|stuffy|blocked|sore\s+throat|throat|sinus\w*)\b", re.IGNORECASE
)
_COUGH_RE = re.compile(r"\bcough\w*\b", re.IGNORECASE)
_DISCLAIMER = "This is general OTC guidance, not a medical diagnosis — consult a doctor if symptoms persist or worsen."

_AGE_WORDS = {
    "child": ("child", "kid", "kids", "baby", "infant", "toddler"),
    "adult": ("adult", "myself", "grown", "me"),
}
_COUGH_TYPE_WORDS = {
    "dry": ("dry",),
    "wet": ("wet", "chesty", "productive", "phlegm", "mucus", "sputum"),
}

_AGE_QUESTION = (
    "Is this for a child, or for an adult/yourself? Our child and adult products use different "
    "dosing and formulations, so this makes sure the recommendation is the right fit."
)
_COUGH_QUESTION = (
    "Is your cough dry, or is it bringing up mucus (a chesty/productive cough)? We carry a different "
    "product for each, so this makes sure you get the right one."
)
_BOTH_QUESTION = (
    "Two quick questions so I recommend the right products: is this for a child, or for an adult/yourself — "
    "and is the cough dry, or is it bringing up mucus (a chesty/productive cough)?"
)
_NO_CHILD_WET_COUGH = (
    "we don't have a product for a child's wet/productive cough — I'd recommend checking with a pharmacist "
    "or pediatrician."
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def _pick(text: str, table: dict[str, tuple]) -> str:
    tokens = _words(text)
    hits = [key for key, words in table.items() if tokens & set(words)]
    return hits[0] if len(hits) == 1 else ""  # both or neither: unknown


def detect_age(text: str) -> str:
    return _pick(text, _AGE_WORDS)


def detect_cough_type(text: str) -> str:
    return _pick(text, _COUGH_TYPE_WORDS)


def groups_in(text: str) -> list[str]:
    """The symptom groups the text names: fever, cough, cold (cold symptoms
    other than the cough itself)."""
    categories = set(tools.classify_categories(text))
    found = []
    if "fever" in categories:
        found.append("fever")
    if _COUGH_RE.search(text):
        found.append("cough")
    if "cold" in categories and _COLD_WORDS_RE.search(text):
        found.append("cold")
    return found


def _format(groups: list[str], age: str, cough: str) -> str:
    return f"{PREFIX}{','.join(groups)};age={age};cough={cough}"


def _parse(trigger: str) -> tuple[list[str], str, str]:
    body = trigger[len(PREFIX):]
    groups_part, _, rest = body.partition(";")
    fields = dict(item.partition("=")[::2] for item in rest.split(";") if "=" in item)
    return groups_part.split(","), fields.get("age", ""), fields.get("cough", "")


def _question(age: str, cough: str) -> str:
    if not age and not cough:
        return _BOTH_QUESTION
    return _AGE_QUESTION if not age else _COUGH_QUESTION


def start(text: str, session_id: str) -> str | None:
    """The reply for a cough-plus-something message (a question, or the
    finished recommendation), or None when this module does not apply. Sets the
    pending clarification when it asks."""
    groups = groups_in(text)
    if "cough" not in groups or len(groups) < 2:
        return None
    age, cough = detect_age(text), detect_cough_type(text)
    return _advance(groups, age, cough, session_id)


def resume(trigger: str, answer: str, session_id: str) -> str | None:
    """Continues after the customer answered. None when the answer supplied
    nothing new (the caller then falls back to a normal turn)."""
    groups, age, cough = _parse(trigger)
    new_age = age or detect_age(answer)
    new_cough = cough or detect_cough_type(answer)
    if (new_age, new_cough) == (age, cough):
        return None
    return _advance(groups, new_age, new_cough, session_id)


def _advance(groups: list[str], age: str, cough: str, session_id: str) -> str:
    if age and cough:
        return _recommend(groups, age, cough, session_id)
    store.set_pending_clarification(session_id, _format(groups, age, cough))
    return _question(age, cough)


def _find(product_ids: list[str]) -> list[dict]:
    return [p for p in (store.find_product(pid) for pid in product_ids) if p]


def _sections(groups: list[str], age: str, cough: str) -> list[tuple[str, list[dict], str]]:
    """(heading, products, note) per symptom."""
    sections = []
    if "fever" in groups:
        sections.append(("For the fever", _find(["fev-004"] if age == "child" else ["fev-001", "fev-002", "fev-003"]), ""))
    cough_label = "dry cough" if cough == "dry" else "chesty (productive) cough"
    if age == "child":
        # The children's syrup is the one product for a child's cough/cold, and it is a dry-cough formulation.
        heading = "For the " + (cough_label if "cold" not in groups else f"{cough_label} and cold symptoms")
        if cough == "dry":
            sections.append((heading, _find(["col-006"]), ""))
        else:
            sections.append((heading, [], _NO_CHILD_WET_COUGH))
        return sections
    sections.append((f"For the {cough_label}", _find(["col-004"] if cough == "dry" else ["col-005"]), ""))
    if "cold" in groups:
        # Cold & Flu Multi-Symptom Relief already contains paracetamol, which
        # the knowledge base says not to double up on — leave it out when
        # fever products are being offered.
        cold_ids = ["col-001", "col-002"] + ([] if "fever" in groups else ["col-003"])
        sections.append(("For the cold symptoms (runny nose, congestion, sneezing)", _find(cold_ids), ""))
    return sections


def _recommend(groups: list[str], age: str, cough: str, session_id: str) -> str:
    lines = ["Here's what I'd suggest, by symptom:", ""]
    shown: list[dict] = []
    for heading, products, note in _sections(groups, age, cough):
        lines.append(f"{heading}:")
        for p in products:
            lines.append(f"- {p['name']} ({p['id']}) — {p['description']}")
            if p not in shown:
                shown.append(p)
        if note:
            lines.append(f"- {note[0].upper()}{note[1:]}")
        lines.append("")
    if shown:
        store.set_last_products(session_id, shown)
        store.clear_last_recommended_product(session_id)
        lines += ["Which one would you like to start with?", ""]
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def is_plan_trigger(trigger: str) -> bool:
    return bool(trigger) and trigger.startswith(PREFIX)
