"""Conversation-level evaluation: scripted multi-turn conversations run
against the REAL, deployed pipeline — the actual /api/chat route (driven
via FastAPI's TestClient, not a shortcut around it) — asserting on
STRUCTURAL reply properties that are robust to the chat model's own
wording variance, rather than exact text.

This fills a gap named explicitly in DEMO_QA_PREP.md's own "if I had more
time" answer: pytest (see tests/) stays fast and deterministic by design,
so it never drives a real multi-turn conversation through main.py's
pending-state dispatch chain (address/email/clarification handling,
bare-selection resolution) together with the LLM's own tool-calling. This
does exactly that — the same mix of deterministic short-circuits and real
LLM calls a live user's conversation actually goes through.

Deliberately separate from pytest: some cases here need genuine LLM
inference (the out-of-scope decline, the RAG citation case) and can take
a while depending on local Ollama load, so this is not run in CI on every
push, the same reasoning as rag_eval.py. Run manually:

    python -m app.conversation_eval

Runs against an ISOLATED temp database (see _use_isolated_db below), not
app/data/app.db — this exercises the real order-placement flow, and must
never overwrite the live demo's real saved address or create bogus orders
under the real demo_user.
"""

import re
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app import agent, main, store

ORDER_ID_RE = re.compile(r"ord-\d{4}")
CITATION_RE = re.compile(r"\(Source: [\w.-]+\.md")


def _use_isolated_db() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="conversation_eval_db_")
    store.DB_PATH = Path(tmp_dir) / "conversation_eval.db"
    store._init_db()


def _contains_any(reply: str, words: list[str]) -> bool:
    lower = reply.lower()
    return any(w.lower() in lower for w in words)


def _no_leaked_tool_name(reply: str) -> bool:
    lower = reply.lower()
    return not any(name in lower for name in agent.TOOL_NAMES)


# Each case is a fresh session, run turn by turn; each turn's checks run
# against that turn's real reply. (name, check_fn) pairs so a failure
# report names exactly what didn't hold, not just "case 2 turn 3 failed."
CONVERSATION_CASES = [
    {
        "name": "cough_clarify_then_order",
        # Fully deterministic end to end (the clarifying-question and
        # order-completion state machine bypasses the LLM entirely for
        # every one of these transitions) — the fast, reliable backbone
        # case that a full order flow still works through main.py's real
        # dispatch chain, not just agent.py in isolation.
        "turns": [
            {
                "text": "I have a dry cough",
                "checks": [
                    ("asks the child/adult clarifying question", lambda r: _contains_any(r, ["child", "adult"])),
                    ("no leaked tool name", _no_leaked_tool_name),
                ],
            },
            {
                "text": "for myself",
                "checks": [
                    ("recommends the real dry-cough product", lambda r: "Cough Suppressant" in r),
                    ("asks to confirm the order", lambda r: "order this" in r.lower()),
                ],
            },
            {
                "text": "yes",
                "checks": [
                    ("asks for a shipping address", lambda r: "shipping address" in r.lower()),
                ],
            },
            {
                "text": "123 Test Ave, Springfield",
                "checks": [
                    (
                        "confirms the order with a real order id",
                        lambda r: "Order confirmed!" in r and ORDER_ID_RE.search(r) is not None,
                    ),
                    ("ships to the address just given", lambda r: "123 Test Ave, Springfield" in r),
                ],
            },
        ],
    },
    {
        "name": "cancel_then_reorder",
        # Real bug this pins, found by this exact case: "please cancel my
        # order" right after an order was placed (with no email on file
        # yet, so pending_email_order_id is set) used to be silently
        # swallowed as "no, skip the confirmation email" — "cancel" is a
        # generic decline word — and the real order was never touched. See
        # main.py's _looks_like_order_cancellation.
        "turns": [
            {"text": "I have a dry cough", "checks": [("asks the clarifying question", lambda r: _contains_any(r, ["child", "adult"]))]},
            {"text": "for myself", "checks": [("recommends the real product", lambda r: "Cough Suppressant" in r)]},
            {"text": "yes", "checks": [("asks for a shipping address", lambda r: "shipping address" in r.lower())]},
            {
                "text": "123 Cancel Reorder Rd",
                "checks": [("first order confirmed", lambda r: "Order confirmed!" in r and "ord-0001" in r)],
            },
            {
                "text": "please cancel my order",
                "checks": [
                    ("actually cancels the real order, not the email offer", lambda r: "cancelled" in r.lower() and "ord-0001" in r),
                ],
            },
            {
                "text": "reorder that for me",
                "checks": [
                    ("confirms the address on file for the same product", lambda r: "Cancel Reorder Rd" in r),
                ],
            },
            {
                "text": "yes",
                "checks": [
                    ("second order confirmed with a new order id", lambda r: "Order confirmed!" in r and "ord-0002" in r),
                    ("same product reordered", lambda r: "Cough Suppressant" in r),
                ],
            },
        ],
    },
    {
        "name": "rejecting_the_address_on_file_for_a_new_one",
        # Real, previously-shipped bugs this shape pins (see main.py's
        # ADDRESS_REJECTION_WORDS comment): several rejection phrasings
        # ("no new one", "i will give another address") used to all hit
        # the same unhelpful "should I ship to X, or a different one?"
        # re-ask verbatim, producing a stuck loop with no progress. The
        # order must also actually ship to the NEW address, not silently
        # keep the old one.
        "turns": [
            {"text": "I have a dry cough", "checks": [("asks the clarifying question", lambda r: _contains_any(r, ["child", "adult"]))]},
            {"text": "for myself", "checks": [("recommends the real product", lambda r: "Cough Suppressant" in r)]},
            {"text": "yes", "checks": [("asks for a shipping address", lambda r: "shipping address" in r.lower())]},
            {
                "text": "123 First Rd",
                "checks": [("first order confirmed", lambda r: "Order confirmed!" in r and "ord-0001" in r)],
            },
            {
                "text": "reorder that",
                "checks": [("confirms the address already on file", lambda r: "123 First Rd" in r)],
            },
            {
                "text": "no, i will give a different one",
                "checks": [
                    ("asks for the new address without repeating the compound question", lambda r: "new shipping address" in r.lower()),
                ],
            },
            {
                "text": "456 Second Ave",
                "checks": [
                    ("second order confirmed", lambda r: "Order confirmed!" in r and "ord-0002" in r),
                    ("ships to the new address, not the old one", lambda r: "456 Second Ave" in r and "123 First Rd" not in r),
                ],
            },
        ],
    },
    {
        "name": "bare_numeric_selection_from_a_product_list",
        # Fully deterministic — proves main.py's _resolve_bare_selection
        # dispatch is actually wired correctly end-to-end through the real
        # endpoint, complementing the unit tests that exercise it in
        # isolation (tests/test_main_heuristics.py).
        "turns": [
            {
                "text": "I have a runny nose, just for myself",
                "checks": [("shows a multi-product list", lambda r: "Which one would you like to try?" in r)],
            },
            {
                "text": "2",
                "checks": [
                    ("selected a product and asks for an address, not 'which one'", lambda r: "shipping address" in r.lower()),
                ],
            },
            {
                "text": "123 Bare Selection Rd",
                "checks": [
                    ("confirms the order for the second listed product", lambda r: "Order confirmed!" in r and "Pseudoephedrine" in r),
                ],
            },
        ],
    },
    {
        "name": "wet_cough_for_a_child_declines_safely",
        # Real, previously-shipped bug (see agent.py's _AGE_WORDS comment):
        # this used to recommend the ADULT expectorant to a child.
        "turns": [
            {
                "text": "my child has a wet cough",
                "checks": [
                    ("declines safely instead of recommending a product", lambda r: _contains_any(r, ["pharmacist", "pediatrician"])),
                    ("never recommends the adult expectorant", lambda r: "guaifenesin" not in r.lower()),
                ],
            },
        ],
    },
    {
        "name": "out_of_scope_decline",
        # Needs the real LLM to decide to call decline_out_of_scope.
        "turns": [
            {
                "text": "can you help me fix a bug in my python script",
                "checks": [
                    ("declines out of scope", lambda r: "can't help with that here" in r.lower()),
                    ("no leaked tool name", _no_leaked_tool_name),
                ],
            },
        ],
    },
    {
        "name": "rag_info_question_with_citation",
        # Needs the real LLM to call lookup_medicine_info and cite it.
        "turns": [
            {
                "text": "dosage for paracetamol 500mg",
                "checks": [
                    ("cites a real knowledge-base source", lambda r: CITATION_RE.search(r) is not None),
                    ("includes the real dosage fact", lambda r: _contains_any(r, ["4-6 hours", "4000mg"])),
                ],
            },
        ],
    },
]


def run_eval() -> tuple[int, int]:
    client = TestClient(main.app)

    passed = 0
    total = 0
    for case in CONVERSATION_CASES:
        # A fresh isolated DB per case, not just once for the whole run —
        # this is a single-demo-user app by design, so without this a
        # later case would inherit the previous case's saved address/
        # orders (real, correct behavior for one shared user across
        # sessions — just not what each case's scripted turns assume
        # starting from a blank slate).
        _use_isolated_db()
        session_id = f"conversation-eval-{case['name']}"
        print(f"\n=== {case['name']} ===")
        for i, turn in enumerate(case["turns"], start=1):
            res = client.post("/api/chat", json={"session_id": session_id, "text": turn["text"]})
            res.raise_for_status()
            reply = res.json()["reply"]
            print(f"  turn {i}: {turn['text']!r}")
            print(f"    reply: {reply[:200]!r}")
            for check_name, check_fn in turn["checks"]:
                total += 1
                ok = bool(check_fn(reply))  # a check accidentally returning e.g. a re.Match must not break the tally
                print(f"    [{'PASS' if ok else 'FAIL'}] {check_name}")
                passed += ok

    print(f"\n{passed}/{total} checks passed")
    return passed, total


if __name__ == "__main__":
    passed, total = run_eval()
    sys.exit(0 if passed == total else 1)
