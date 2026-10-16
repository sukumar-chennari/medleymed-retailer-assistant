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
