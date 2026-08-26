"""A small RAG evaluation harness, checked at TWO separate layers:

1. Retrieval — does retrieval.search() itself return the right document,
   with the right facts present across the top-k chunks?
2. Agent — does the actual deployed chat pipeline (agent.run_turn, the
   same code /api/chat calls) produce a reply containing those facts?

These can genuinely diverge, and testing only the first one is a real gap,
not a theoretical one: this project shipped a routing bug where
"dosage for paracetamol 500mg" scored the correct chunk every time at the
retrieval layer, while the agent itself misrouted the message into a
clarifying question and never called lookup_medicine_info at all — a
retrieval-only report would have stayed green throughout. Layer 2 is what
actually answers "does the assistant give this answer," and is the one
that matters to a user; layer 1 stays useful for isolating *which* layer
broke when something fails.

Run standalone with `python -m app.rag_eval` (runs both layers) or
`python -m app.rag_eval --fast` (retrieval only, skips the slow real LLM
calls — useful while iterating on retrieval.py itself).
"""

import html
import re
import sys
from pathlib import Path

from app import agent, retrieval, store

REPORT_PATH = Path(__file__).parent / "data" / "rag_eval_report.html"

# expected_source=None means "should retrieve nothing" — these guard the
# scope boundary that keeps RAG from becoming a backdoor for unrelated
# questions (see agent.py's MEDICINE INFO / SCOPE instructions). Every
# in-scope case's golden_answer is a reference answer taken directly from
# the corresponding knowledge_base/*.md file, and must_include is the
# minimal set of facts a *correct* answer has to actually surface — checked
# against both the retrieved chunks (layer 1) and the real agent reply
# (layer 2).
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
        "query": "dosage for guaifenesin expectorant syrup",
        "expected_source": "col-005.md",
        "expected_section": "Dosage",
        "golden_answer": (
            "Adults typically take the dose listed on the label (commonly "
            "200-400mg) every 4 hours as needed, not exceeding 6 doses "
            "(2400mg) in 24 hours."
        ),
        "must_include": ["200-400mg", "4 hours"],
    },
    {
        "query": "dosing for children's cold and cough syrup",
        "expected_source": "col-006.md",
        "expected_section": "Dosage",
        "golden_answer": (
            "Dosage is based on weight and age, not a flat adult dose — use "
            "the dosing cup or syringe provided and follow the weight/age "
            "chart on the label exactly."
        ),
        "must_include": ["weight", "age"],
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


def _evaluate_retrieval(case: dict, top_k: int) -> dict:
    """Layer 1: runs one case through retrieval.search() directly and
    scores it against hit@k and content-grounding. Content grounding is
    checked across *all* top_k results combined, not just the rank-1
    chunk — that's what lookup_medicine_info actually hands the model."""
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
        "results": results,
        "source_hit": source_hit,
        "content_hit": content_hit,
        "retrieval_passed": source_hit and content_hit,
        "retrieval_detail": detail,
    }


_MG_RE = re.compile(r"\d+\s*mg", re.IGNORECASE)


def _looks_fabricated(reply: str) -> str | None:
    """For an out-of-scope query, the agent should decline or say it's not
    in the knowledge base — never invent a specific dosage number or name
    one of our real catalog products (which would mean it got misrouted
    into a real recommendation instead of declining). Returns a reason
    string if the reply looks fabricated, else None."""
    if _MG_RE.search(reply):
        return "reply contains a fabricated-looking dosage number (Nmg)"
    catalog_names = [p["name"].lower() for p in store.get_catalog()]
    mentioned = [n for n in catalog_names if n in reply.lower()]
    if mentioned:
        return f"reply names a real catalog product: {mentioned}"
    return None


def _evaluate_agent(case: dict, index: int) -> dict:
    """Layer 2: runs one case through the REAL agent pipeline
    (agent.run_turn — the same function /api/chat calls), with a fresh
    session per case so no pending state or history from one case leaks
    into the next. This is the check that answers "does the assistant
    actually say this," not just "does retrieval find the right text" —
    the two are verifiably not the same claim (see module docstring)."""
    session_id = f"rag-agent-eval-{index}"
    reply, _ = agent.run_turn([], user_text=case["query"], session_id=session_id)

    if case["golden_answer"] is None:
        fabrication = _looks_fabricated(reply)
        agent_passed = fabrication is None
        detail = "correctly declined / no fabrication" if agent_passed else fabrication
    else:
        missing = [f for f in case["must_include"] if f.lower() not in reply.lower()]
        agent_passed = not missing
        detail = "reply contains the golden facts" if agent_passed else f"reply is missing: {missing}"

    return {"agent_reply": reply, "agent_passed": agent_passed, "agent_detail": detail}


def run_eval(top_k: int = 3, include_agent: bool = True) -> tuple[int, int]:
    evaluated = []
    for i, case in enumerate(EVAL_CASES):
        row = {**case, **_evaluate_retrieval(case, top_k)}
        status = "PASS" if row["retrieval_passed"] else "FAIL"
        print(f"[retrieval {status}] {case['query']!r} — {row['retrieval_detail']}")

        if include_agent:
            row.update(_evaluate_agent(case, i))
            status = "PASS" if row["agent_passed"] else "FAIL"
            print(f"[agent     {status}] {case['query']!r} — {row['agent_detail']}")
        evaluated.append(row)

    build_html_report(evaluated, include_agent)
    print(f"\nHTML report written to {REPORT_PATH}")

    retrieval_passed = sum(r["retrieval_passed"] for r in evaluated)
    if include_agent:
        agent_passed = sum(r["agent_passed"] for r in evaluated)
        print(f"Retrieval layer: {retrieval_passed}/{len(evaluated)} passed")
        print(f"Agent layer:     {agent_passed}/{len(evaluated)} passed")
        return agent_passed, len(evaluated)
    return retrieval_passed, len(evaluated)


def build_html_report(evaluated: list[dict], include_agent: bool) -> None:
    """Writes a single self-contained HTML file — query / golden answer /
    retrieved chunks / real agent reply, side by side, with retrieval and
    agent pass/fail shown as SEPARATE badges. Keeping them separate (not
    collapsed into one pass/fail bit) is the whole point: a query that
    passes retrieval but fails at the agent layer is a routing bug, not a
    retrieval bug, and the report should make that distinction visible
    instead of just saying "fail" and leaving the reader to guess why."""
    rows = []
    for r in evaluated:
        results = r["results"]
        if r["golden_answer"] is None:
            golden_cell = "<em>(out of scope — should decline / not fabricate)</em>"
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

        agent_cell = ""
        if include_agent:
            agent_ok = r["agent_passed"]
            checks += (
                f" <span class=\"check {'ok' if agent_ok else 'bad'}\">"
                f"{'✓' if agent_ok else '✗'} agent</span>"
            )
            agent_cell = (
                f"<div class=\"meta\">real /api/chat reply &mdash; "
                f"{'passed' if agent_ok else html.escape(r['agent_detail'])}</div>"
                f"<div class=\"chunk\">{html.escape(r['agent_reply'])}</div>"
            )

        overall_pass = r["retrieval_passed"] and (not include_agent or r["agent_passed"])
        badge = "pass" if overall_pass else "fail"

        rows.append(f"""
        <tr class="{badge}">
          <td class="query">{html.escape(r['query'])}</td>
          <td>{golden_cell}</td>
          <td>{retrieved_cell}</td>
          {"<td>" + agent_cell + "</td>" if include_agent else ""}
          <td class="checks">{checks}</td>
        </tr>""")

    retrieval_passed = sum(r["retrieval_passed"] for r in evaluated)
    total = len(evaluated)
    if include_agent:
        agent_passed = sum(r["agent_passed"] for r in evaluated)
        summary = (
            f"Retrieval layer: <span class=\"score\">{retrieval_passed}/{total}</span> &mdash; "
            f"Agent layer (real /api/chat replies): <span class=\"score\">{agent_passed}/{total}</span>. "
            f"Retrieval passing doesn't imply the agent layer does — they're checked and shown separately "
            f"on purpose, since a routing bug can make one pass while the other fails."
        )
    else:
        summary = (
            f"Retrieval layer only: <span class=\"score\">{retrieval_passed}/{total}</span> "
            f"&mdash; run without --fast to also verify the real agent replies."
        )

    agent_header = "<th>Real Agent Reply</th>" if include_agent else ""

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RAG Golden Eval Report</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a2e; background: #fafafa; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .summary {{ font-size: 1.05rem; margin-bottom: 1.5rem; max-width: 80ch; }}
  .summary .score {{ font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  th, td {{ border: 1px solid #e2e2e8; padding: 0.75rem; vertical-align: top; text-align: left; font-size: 0.92rem; }}
  th {{ background: #2d3480; color: #fff; position: sticky; top: 0; }}
  tr.pass {{ background: #f3fbf5; }}
  tr.fail {{ background: #fdf3f3; }}
  .query {{ font-weight: 600; width: 14%; }}
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
  <div class="summary">{summary}</div>
  <table>
    <thead>
      <tr><th>Query</th><th>Golden Answer</th><th>Retrieved Chunks</th>{agent_header}<th>Checks</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>"""

    REPORT_PATH.write_text(html_doc)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    passed, total = run_eval(include_agent=not fast)
    print()
    label = "Retrieval-only" if fast else "Agent-level"
    print(f"{label} result: {passed}/{total} passed ({passed / total:.0%})")
