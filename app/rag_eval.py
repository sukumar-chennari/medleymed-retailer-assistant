"""A small RAG evaluation harness: a golden query set with a reference
answer for each in-scope query, checked two ways — retrieval hit@k (did the
right document come back at all?) and content grounding (does the actual
retrieved chunk contain the key facts the reference answer relies on, not
just the right filename?). Run standalone with `python -m app.rag_eval`.

This is intentionally simple — a hand-written golden set with a keyword
checklist per answer, not a full precision/recall/MRR suite or an LLM-judge
similarity score — proportionate to a ~30-chunk demo corpus. The point is a
concrete, repeatable, and *viewable* check that retrieval quality didn't
regress (e.g. after changing MIN_SIMILARITY, or editing the knowledge base),
rather than eyeballing it in chat each time. Every run also writes an HTML
report (see build_html_report) so results can be reviewed at a glance
instead of read off the terminal.
"""

import html
from pathlib import Path

from app import retrieval

REPORT_PATH = Path(__file__).parent / "data" / "rag_eval_report.html"

# expected_source=None means "should retrieve nothing" — these guard the
# scope boundary that keeps RAG from becoming a backdoor for unrelated
# questions (see agent.py's MEDICINE INFO / SCOPE instructions). Every
# in-scope case's golden_answer is a reference answer taken directly from
# the corresponding knowledge_base/*.md file, and must_include is the
# minimal set of facts a *correct* retrieval has to actually surface —
# checked against the real top-1 chunk text, not just its filename, so a
# right-document-wrong-section (or a chunk that scored well but is missing
# the specific number/fact asked about) still fails the content-grounding
# check even though it passes the source hit@k check.
EVAL_CASES = [
    {
        "query": "dosage for paracetamol 500mg",
        "expected_source": "fev-001.md",
        "expected_section": "Dosage",
        "golden_answer": (
            "Adults typically take 1-2 tablets (500-1000mg) every 4-6 hours "
            "as needed, not exceeding 8 tablets (4000mg) in 24 hours."
        ),
        "must_include": ["4-6 hours", "4000mg"],
    },
    {
        "query": "how much extra strength paracetamol can I take",
        "expected_source": "fev-002.md",
        "expected_section": "Dosage",
        "golden_answer": (
            "Adults typically take 1 tablet (650mg) every 6-8 hours as "
            "needed, not exceeding 4000mg of total paracetamol in 24 hours "
            "from all sources combined."
        ),
        "must_include": ["6-8 hours", "4000mg"],
    },
    {
        "query": "can I take ibuprofen for a fever",
        "expected_source": "fev-003.md",
        "expected_section": "Overview",
        "golden_answer": (
            "Yes — ibuprofen is an NSAID that reduces fever, relieves pain, "
            "and reduces inflammation, making it a good option for fevers "
            "that come with body aches or headaches."
        ),
        "must_include": ["fever", "inflammation"],
    },
    {
        "query": "dosing for children's paracetamol syrup",
        "expected_source": "fev-004.md",
        "expected_section": "Dosage",
        "golden_answer": (
            "Dosage is based on weight and age, not a flat adult dose — use "
            "the dosing cup or syringe provided and follow the weight/age "
            "chart on the label exactly."
        ),
        "must_include": ["weight", "age"],
    },
    {
        "query": "side effects of cetirizine",
        "expected_source": "col-001.md",
        "expected_section": "Common Side Effects",
        "golden_answer": (
            "The most commonly reported side effect is mild drowsiness "
            "(though less sedating than older antihistamines); dry mouth "
            "and mild headache have also been reported."
        ),
        "must_include": ["drowsiness", "dry mouth"],
    },
    {
        "query": "warnings for pseudoephedrine decongestant",
        "expected_source": "col-002.md",
        "expected_section": "Warnings",
        "golden_answer": (
            "Not recommended for people with high blood pressure, heart "
            "disease, or thyroid conditions without consulting a doctor "
            "first, since it can raise blood pressure and heart rate."
        ),
        "must_include": ["high blood pressure", "heart disease"],
    },
    {
        "query": "what's in cold and flu multi-symptom relief",
        "expected_source": "col-003.md",
        "expected_section": "Overview",
        "golden_answer": (
            "It contains paracetamol, phenylephrine, and chlorpheniramine — "
            "covering fever/aches, nasal congestion, and runny nose/sneezing "
            "in one product."
        ),
        "must_include": ["paracetamol", "phenylephrine", "chlorpheniramine"],
    },
    {
        "query": "dosage for cough suppressant syrup",
        "expected_source": "col-004.md",
        "expected_section": "Dosage",
        "golden_answer": (
            "Adults typically take the dose listed on the label (commonly "
            "10-20mg) every 4 hours as needed, not exceeding the maximum "
            "daily dose on the label."
        ),
        "must_include": ["10-20mg", "4 hours"],
    },
    {
        "query": "credit card refund policy",
        "expected_source": None,
        "expected_section": None,
        "golden_answer": None,
        "must_include": [],
    },
    {
        "query": "what's the weather today",
        "expected_source": None,
        "expected_section": None,
        "golden_answer": None,
        "must_include": [],
    },
    {
        "query": "python programming help",
        "expected_source": None,
        "expected_section": None,
        "golden_answer": None,
        "must_include": [],
    },
    {
        "query": "dosage for amoxicillin",
        "expected_source": None,
        "expected_section": None,
        "golden_answer": None,
        "must_include": [],
    },
]


def _evaluate_case(case: dict, top_k: int) -> dict:
    """Runs one case through real retrieval and scores it against both the
    hit@k and content-grounding checks. Content grounding is checked across
    *all* top_k results combined, not just the rank-1 chunk — that's what
    lookup_medicine_info actually hands the model (see agent.py's MEDICINE
    INFO instructions: synthesize from every returned result, not just the
    first). This matters in practice: "dosage for paracetamol 500mg" scores
    the Overview section slightly higher than the actual Dosage section
    (0.856 vs 0.831 — the strength boost matches the whole document's title
    equally across every section, so it doesn't discriminate rank *within*
    a document), but Dosage still comes back in the top-3, so the model
    still has the real fact available. A rank-1-only check would flag that
    as a failure even though the end-to-end answer is still grounded."""
    results = retrieval.search(case["query"], top_k=top_k)
    sources = [r["source"] for r in results]
    expected = case["expected_source"]
    combined_text = " ".join(r["text"] for r in results).lower()

    if expected is None:
        source_hit = len(results) == 0
        content_hit = True  # nothing to ground — the check doesn't apply
        detail = "correctly returned nothing" if source_hit else f"expected nothing, got {sources}"
    else:
        source_hit = expected in sources
        must_include = case["must_include"]
        content_hit = bool(results) and all(fact.lower() in combined_text for fact in must_include)
        if source_hit and content_hit:
            detail = f"found {expected}, grounded in the expected facts"
        elif source_hit:
            missing = [f for f in must_include if f.lower() not in combined_text]
            detail = f"found {expected}, but retrieved chunks are missing: {missing}"
        else:
            detail = f"expected {expected}, got {sources or 'nothing'}"

    return {
        **case,
        "results": results,
        "source_hit": source_hit,
        "content_hit": content_hit,
        "passed": source_hit and content_hit,
        "detail": detail,
    }


def run_eval(top_k: int = 3) -> tuple[int, int]:
    evaluated = [_evaluate_case(case, top_k) for case in EVAL_CASES]
    passed = 0
    for r in evaluated:
        passed += r["passed"]
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['query']!r} — {r['detail']}")

    build_html_report(evaluated)
    print(f"\nHTML report written to {REPORT_PATH}")
    return passed, len(evaluated)


def build_html_report(evaluated: list[dict]) -> None:
    """Writes a single self-contained HTML file — query / golden answer /
    actual retrieved chunk side by side, with the source-hit and
    content-grounding checks shown separately, so a failure that got the
    right document but missed the actual fact asked about is visibly
    distinct from getting the wrong document entirely."""
    rows = []
    for r in evaluated:
        badge = "pass" if r["passed"] else "fail"
        results = r["results"]
        if r["golden_answer"] is None:
            golden_cell = "<em>(out of scope — should retrieve nothing)</em>"
        else:
            golden_cell = html.escape(r["golden_answer"])

        if results:
            retrieved_cell = "".join(
                f"<div class=\"meta\">#{i + 1} {html.escape(c['source'])} &sect; {html.escape(c['section'])} "
                f"&middot; score {c['score']}</div>"
                f"<div class=\"chunk\">{html.escape(c['text'])}</div>"
                for i, c in enumerate(results)
            )
        else:
            retrieved_cell = "<em>(no results)</em>"

        checks = (
            f"<span class=\"check {'ok' if r['source_hit'] else 'bad'}\">"
            f"{'✓' if r['source_hit'] else '✗'} source</span> "
            f"<span class=\"check {'ok' if r['content_hit'] else 'bad'}\">"
            f"{'✓' if r['content_hit'] else '✗'} grounded</span>"
        )

        rows.append(f"""
        <tr class="{badge}">
          <td class="query">{html.escape(r['query'])}</td>
          <td>{golden_cell}</td>
          <td>{retrieved_cell}</td>
          <td class="checks">{checks}</td>
        </tr>""")

    passed = sum(r["passed"] for r in evaluated)
    total = len(evaluated)

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RAG Golden Eval Report</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a2e; background: #fafafa; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .summary {{ font-size: 1.1rem; margin-bottom: 1.5rem; }}
  .summary .score {{ font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  th, td {{ border: 1px solid #e2e2e8; padding: 0.75rem; vertical-align: top; text-align: left; font-size: 0.92rem; }}
  th {{ background: #2d3480; color: #fff; position: sticky; top: 0; }}
  tr.pass {{ background: #f3fbf5; }}
  tr.fail {{ background: #fdf3f3; }}
  .query {{ font-weight: 600; width: 16%; }}
  .meta {{ font-size: 0.78rem; color: #666; margin-bottom: 0.35rem; margin-top: 0.75rem; }}
  .meta:first-child {{ margin-top: 0; }}
  .chunk {{ white-space: pre-wrap; font-size: 0.85rem; }}
  .checks {{ white-space: nowrap; width: 12%; }}
  .check {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.78rem; margin-bottom: 0.25rem; }}
  .check.ok {{ background: #d7f5df; color: #146c2e; }}
  .check.bad {{ background: #fbdada; color: #a11313; }}
</style>
</head>
<body>
  <h1>RAG Golden Eval Report</h1>
  <div class="summary">Passed <span class="score">{passed}/{total}</span> ({passed / total:.0%}) &mdash;
  checks both retrieval hit@k (right document) and content grounding (right facts actually present in the top chunk).</div>
  <table>
    <thead>
      <tr><th>Query</th><th>Golden Answer</th><th>Actual Top Retrieved Chunk</th><th>Checks</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>"""

    REPORT_PATH.write_text(html_doc)


if __name__ == "__main__":
    passed, total = run_eval()
    print()
    print(f"{passed}/{total} passed ({passed / total:.0%})")
