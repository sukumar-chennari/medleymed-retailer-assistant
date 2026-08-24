"""A small RAG evaluation harness: a fixed set of test queries with the
source document each one should (or, for out-of-scope queries, should NOT)
retrieve. Run standalone with `python -m app.rag_eval`.

This is intentionally simple — hit@k against a hand-written test set, not a
full precision/recall/MRR suite — proportionate to a ~30-chunk demo corpus.
The point is to have *some* concrete, repeatable check that retrieval quality
didn't regress (e.g. after changing MIN_SIMILARITY, or editing the knowledge
base), rather than eyeballing it in chat each time.
"""

from app import retrieval

# expected_source=None means "should retrieve nothing" — these guard the
# scope boundary that keeps RAG from becoming a backdoor for unrelated
# questions (see agent.py's MEDICINE INFO / SCOPE instructions).
EVAL_CASES = [
    {"query": "dosage for paracetamol 500mg", "expected_source": "fev-001.md"},
    {"query": "how much extra strength paracetamol can I take", "expected_source": "fev-002.md"},
    {"query": "can I take ibuprofen for a fever", "expected_source": "fev-003.md"},
    {"query": "dosing for children's paracetamol syrup", "expected_source": "fev-004.md"},
    {"query": "side effects of cetirizine", "expected_source": "col-001.md"},
    {"query": "warnings for pseudoephedrine decongestant", "expected_source": "col-002.md"},
    {"query": "what's in cold and flu multi-symptom relief", "expected_source": "col-003.md"},
    {"query": "dosage for cough suppressant syrup", "expected_source": "col-004.md"},
    {"query": "credit card refund policy", "expected_source": None},
    {"query": "what's the weather today", "expected_source": None},
    {"query": "python programming help", "expected_source": None},
    {"query": "dosage for amoxicillin", "expected_source": None},
]


def run_eval(top_k: int = 3) -> tuple[int, int]:
    passed = 0
    for case in EVAL_CASES:
        results = retrieval.search(case["query"], top_k=top_k)
        sources = [r["source"] for r in results]
        expected = case["expected_source"]

        if expected is None:
            ok = len(results) == 0
            detail = "correctly returned nothing" if ok else f"expected nothing, got {sources}"
        else:
            ok = expected in sources
            detail = f"found {expected}" if ok else f"expected {expected}, got {sources or 'nothing'}"

        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['query']!r} — {detail}")

    return passed, len(EVAL_CASES)


if __name__ == "__main__":
    passed, total = run_eval()
    print()
    print(f"{passed}/{total} passed ({passed / total:.0%})")
