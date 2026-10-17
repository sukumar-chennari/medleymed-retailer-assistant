"""Safety topics a customer may ask about a medicine (pregnancy, alcohol, ...)
and where the knowledge base would state them. Dependency-free on purpose: both
retrieval.py (ranking) and guardrails.py (grounding check) use it, and they
cannot import each other."""

import re

# Safety topics a customer may ask about a medicine, with where such a fact
# would appear in the knowledge base. Real bug (Oct 16 live probe): the
# retriever always returns its nearest chunks (score ~0.65 even when they say
# nothing relevant) and the model filled the gap from general knowledge while
# attributing it to us — "According to our knowledge base, cetirizine is
# generally considered..." about pregnancy when col-001.md never mentions
# pregnancy, "paracetamol typically starts working within 30 minutes" (no such
# text), storage advice for a syrup with no storage text anywhere. For a
# safety question that is the worst place to improvise.
_SAFETY_TOPICS = (
    ("pregnancy", r"pregnan", r"pregnan"),
    ("breastfeeding", r"breast-?\s?feed|nursing mother|lactat", r"breast|nursing|lactat"),
    ("driving", r"\bdriv\w*", r"driv|operat\w* (?:machin|heavy)"),
    ("alcohol", r"alcohol", r"alcohol"),
    ("storage", r"\bstor(?:e|age|ing)\b|shelf life|expir\w+", r"stor(?:e|age)|room temperature|cool|dry place|expir"),
    ("asthma", r"asthma", r"asthma"),
    ("diabetes", r"diabet", r"diabet"),
    ("kidney problems", r"kidney|renal", r"kidney|renal"),
    ("liver problems", r"\bliver\b|hepat", r"liver|hepat"),
    ("blood pressure", r"blood pressure|hypertens", r"blood pressure|hypertens"),
    ("stomach ulcers", r"\bulcers?\b", r"ulcer"),
    # Product facts the knowledge base has no text for at all. Found Oct 17:
    # "how many tablets in a pack of paracetamol" got "usually around 20-30
    # tablets per pack" and "how long does paracetamol take to work" got
    # "30 minutes to an hour ... according to our knowledge base" — both
    # invented, both attributed to us. The "covered" patterns avoid bare
    # "N tablets", which every dosage chunk contains.
    ("pack size",
     r"pack\s+size|per\s+(?:pack|box|bottle|strip)|(?:tablets|pills|capsules|ml)\s+(?:in|per|come\s+in)\s+(?:a\s+|each\s+|the\s+|one\s+)?(?:pack|box|bottle|strip)"
     r"|how\s+many\s+\w+\s+(?:are\s+)?(?:in|come\s+in)\s+(?:a|each|the|one)\s+(?:pack|box|bottle|strip)|(?:pack|box|bottle)\s+(?:size|contain\w*)|how\s+big\s+is\s+(?:a|the)\s+(?:pack|bottle)",
     r"\bpacks?\b|\bpackage\b|per\s+(?:box|bottle|strip)|bottle\s+(?:size|contains?)|contains?\s+\d+\s+(?:tablets|capsules|ml)|\d+\s?ml\s+(?:bottle|pack)"),
    ("how quickly it works",
     r"how\s+(?:long|quickly|fast|soon)\b.{0,30}(?:work|take\s+effect|kick\s+in|start|begin|relief|feel\s+better)|\bonset\b|time\s+to\s+(?:work|take\s+effect)",
     r"\bonset\b|takes?\s+effect|start(?:s)?\s+(?:to\s+)?work|within\s+\d+\s*(?:minutes?|mins?|hours?)|\d+\s*(?:minutes?|mins?)\s+(?:to|after)|kick"),
    ("missed doses", r"(?:missed|miss|forgot)\s+(?:a\s+|my\s+)?dose|forgot to take", r"miss|forgot|skip"),
)

_COMPILED = tuple(
    (label, re.compile(asked, re.IGNORECASE), re.compile(covered, re.IGNORECASE))
    for label, asked, covered in _SAFETY_TOPICS
)


def topics_asked(question: str) -> list[str]:
    return [label for label, asked, _ in _COMPILED if asked.search(question)]


def covered_patterns_for(question: str) -> list:
    """Compiled patterns that find, in a chunk, the facts the question's
    safety topics ask for (used by retrieval to rank those chunks)."""
    return [covered for _, asked, covered in _COMPILED if asked.search(question)]


def uncovered(question: str, results: list[dict]) -> list[str]:
    """The safety topics the question asks about that NONE of the retrieved
    knowledge-base text mentions."""
    corpus = " ".join(str(r.get("text", "")) for r in results)
    return [
        label for label, asked, covered in _COMPILED
        if asked.search(question) and not covered.search(corpus)
    ]
