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


# Reply-text checks alone proved too weak: on Oct 7 a run scored the model's
# INVENTED receipt ("Order ID: ord-0002 / Product: ... / Shipping to: 123
# First Rd" — it had skipped the reorder_last tool and no order existed) as a
# PASS for "confirms the address already on file", because the check only
# looked for the address string anywhere in the reply. This harness runs
# in-process against an isolated database, so turns can also assert on REAL
# state (turn["state_checks"], each called with the session id) — a made-up
# reply can't fake an order row.
def _asks_to_confirm_address(address: str):
    """The deterministic deferred-order reply (guardrails.reply_for_deferred_
    order): "We have this address on file: X. Should I ship to this address,
    or would you like to give a different one?" — and NOT a receipt."""
    return lambda r: address in r and "address on file" in r.lower() and "Order ID" not in r


def _order_count_is(expected: int):
    return lambda session_id: len(store.list_orders("demo_user")) == expected


def _order_status_is(order_id: str, status: str):
    return lambda session_id: (store.get_order(order_id) or {}).get("status") == status


def _awaiting_address_confirmation(session_id: str) -> bool:
    return store.get_pending_address_confirmation(session_id) is not None


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
                "state_checks": [("a real order exists", _order_count_is(1))],
            },
            {
                "text": "please cancel my order",
                "state_checks": [("the real order is actually cancelled", _order_status_is("ord-0001", "cancelled"))],
                "checks": [
                    ("actually cancels the real order, not the email offer", lambda r: "cancelled" in r.lower() and "ord-0001" in r),
                ],
            },
            {
                "text": "reorder that for me",
                "checks": [
                    ("confirms the address on file for the same product", _asks_to_confirm_address("Cancel Reorder Rd")),
                ],
                "state_checks": [
                    ("no second order exists yet", _order_count_is(1)),
                    ("is genuinely awaiting address confirmation", _awaiting_address_confirmation),
                ],
            },
            {
                "text": "yes",
                "state_checks": [("a real second order exists", _order_count_is(2))],
                "checks": [
                    ("second order confirmed with a new order id", lambda r: "Order confirmed!" in r and "ord-0002" in r),
                    ("same product reordered", lambda r: "Cough Suppressant" in r),
                ],
            },
        ],
    },
    {
        "name": "email_after_cancel_declines_to_confirm",
        # Real bug this pins: nothing cleared pending_email when the order
        # was cancelled before the email arrived — supplying the email
        # afterward used to cheerfully claim "I've sent the confirmation"
        # for an order that no longer exists. See main.py's
        # _complete_pending_email.
        "turns": [
            {"text": "I have a dry cough", "checks": [("asks the clarifying question", lambda r: _contains_any(r, ["child", "adult"]))]},
            {"text": "for myself", "checks": [("recommends the real product", lambda r: "Cough Suppressant" in r)]},
            {"text": "yes", "checks": [("asks for a shipping address", lambda r: "shipping address" in r.lower())]},
            {
                "text": "123 Email After Cancel Rd",
                "checks": [
                    ("order confirmed with no email on file", lambda r: "Order confirmed!" in r and "ord-0001" in r),
                    ("asks for an email", lambda r: "email" in r.lower()),
                ],
            },
            {
                "text": "please cancel my order",
                "state_checks": [("the real order is actually cancelled", _order_status_is("ord-0001", "cancelled"))],
                "checks": [("actually cancels the real order", lambda r: "cancelled" in r.lower() and "ord-0001" in r)],
            },
            {
                "text": "email@example.com",
                "checks": [
                    ("declines to confirm the cancelled order", lambda r: "cancelled" in r.lower()),
                    ("never claims a confirmation was sent", lambda r: "sent the confirmation" not in r.lower()),
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
                "state_checks": [("a real order exists", _order_count_is(1))],
                "checks": [("first order confirmed", lambda r: "Order confirmed!" in r and "ord-0001" in r)],
            },
            {
                "text": "reorder that",
                "checks": [("confirms the address already on file", _asks_to_confirm_address("123 First Rd"))],
                "state_checks": [
                    ("no second order exists yet", _order_count_is(1)),
                    ("is genuinely awaiting address confirmation", _awaiting_address_confirmation),
                ],
            },
            {
                "text": "no, i will give a different one",
                "checks": [
                    ("asks for the new address without repeating the compound question", lambda r: "new shipping address" in r.lower()),
                ],
            },
            {
                "text": "456 Second Ave",
                "state_checks": [("a real second order exists", _order_count_is(2))],
                "checks": [
                    ("second order confirmed", lambda r: "Order confirmed!" in r and "ord-0002" in r),
                    ("ships to the new address, not the old one", lambda r: "456 Second Ave" in r and "123 First Rd" not in r),
                ],
            },
        ],
    },
    {
        "name": "unrelated_question_during_address_confirmation_does_not_trap",
        # Real bug this pins: pending_address_confirmation's catch-all used
        # to unconditionally repeat "Just to confirm — should I ship to
        # X..." for ANY reply that wasn't yes/an address/"different
        # address" — even a completely unrelated message — trapping the
        # conversation. Same trapping bug already fixed for
        # pending_product_id/pending_email_order_id (see cancel_then_
        # reorder/out_of_scope_decline), just missing from this third
        # pending-state branch. See main.py's chat() dispatch chain.
        "turns": [
            {"text": "I have a dry cough", "checks": [("asks the clarifying question", lambda r: _contains_any(r, ["child", "adult"]))]},
            {"text": "for myself", "checks": [("recommends the real product", lambda r: "Cough Suppressant" in r)]},
            {"text": "yes", "checks": [("asks for a shipping address", lambda r: "shipping address" in r.lower())]},
            {
                "text": "123 Unrelated Question Rd",
                "checks": [("first order confirmed", lambda r: "Order confirmed!" in r and "ord-0001" in r)],
            },
            {
                "text": "reorder that",
                "checks": [("confirms the address already on file", _asks_to_confirm_address("123 Unrelated Question Rd"))],
                "state_checks": [
                    ("no second order exists yet", _order_count_is(1)),
                    ("is genuinely awaiting address confirmation", _awaiting_address_confirmation),
                ],
            },
            {
                "text": "can you help me fix a bug in my python script",
                "state_checks": [
                    ("no order was invented by the unrelated turn", _order_count_is(1)),
                    ("the confirmation is still pending", _awaiting_address_confirmation),
                ],
                "checks": [
                    ("answers instead of repeating the confirm-address question", lambda r: "can't help with that here" in r.lower()),
                    ("never claims an order was placed", lambda r: "Order confirmed!" not in r),
                ],
            },
            {
                "text": "yes",
                "state_checks": [("a real second order exists", _order_count_is(2))],
                "checks": [
                    ("the pending order still completes afterward", lambda r: "Order confirmed!" in r and "ord-0002" in r),
                ],
            },
        ],
    },
    {
        "name": "ask_assistant_button_names_a_product",
        # Fully deterministic. The dashboard's "Ask Assistant" button sends
        # "I'd like to order <product name>"; on Oct 13 a probe found 0 of 10
        # catalog products started an order from it (the model asked an
        # age/dry-or-wet question or claimed the product wasn't stocked).
        # Naming a different product while one awaits its address must also
        # switch the order, not save the sentence as the shipping address.
        "turns": [
            {
                "text": "I'd like to order Ibuprofen 200mg Tablets",
                "checks": [("asks for a shipping address", lambda r: "address" in r.lower())],
                "state_checks": [("no order placed yet", _order_count_is(0))],
            },
            {
                "text": "I'd like to order Guaifenesin Expectorant Syrup",
                "checks": [("asks for a shipping address again", lambda r: "address" in r.lower())],
                "state_checks": [("the sentence was not taken as an address", _order_count_is(0))],
            },
            {
                "text": "45 Test Road, Pune",
                "checks": [
                    ("orders the product named last", lambda r: "Order confirmed!" in r and "Guaifenesin" in r),
                    ("ships to the address given", lambda r: "45 Test Road, Pune" in r),
                ],
                "state_checks": [("exactly one real order exists", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "named_order_with_a_quantity",
        # Fully deterministic. "order 2 <product>" used to reach the model,
        # which re-asked the age question or invented a product (0 of 5
        # phrasings reached a pending order on Oct 14).
        "turns": [
            {
                "text": "order 2 Cetirizine 10mg Antihistamine Tablets",
                "checks": [("asks for a shipping address", lambda r: "address" in r.lower())],
                "state_checks": [("no order placed yet", _order_count_is(0))],
            },
            {
                "text": "8 Quantity Lane, Pune",
                "checks": [
                    ("confirms the named product", lambda r: "Order confirmed!" in r and "Cetirizine" in r),
                    ("with the quantity asked for", lambda r: "Quantity: 2" in r),
                ],
                "state_checks": [("exactly one real order exists", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "change_and_read_back_the_saved_address",
        # Fully deterministic. On Oct 14 a probe found "change my shipping
        # address to X" made the model invent an "Order updated!" receipt
        # (blocked, then replaced with "what symptom is this for?"), and
        # "what address do you have on file?" got the same non-answer.
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {
                "text": "10 High St, Leeds",
                "checks": [("order placed", lambda r: "Order confirmed!" in r)],
                "state_checks": [("one real order", _order_count_is(1))],
            },
            {
                "text": "change my shipping address to 99 New Rd, York",
                "checks": [
                    ("confirms the new address for future orders", lambda r: "99 New Rd, York" in r and "future orders" in r),
                    ("says the existing order keeps its address", lambda r: "already placed" in r),
                    ("no invented receipt", lambda r: "Order ID" not in r and "symptom" not in r.lower()),
                ],
                "state_checks": [
                    ("no order was created or edited", _order_count_is(1)),
                    ("the old order still ships to the old address", lambda sid: store.get_order("ord-0001")["address"] == "10 High St, Leeds"),
                ],
            },
            {
                "text": "what address do you have on file for me?",
                "checks": [("reads back the new saved address", lambda r: "99 New Rd, York" in r and "symptom" not in r.lower())],
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
        "name": "combined_fever_and_cold_symptoms",
        # Real bug this pins: fever is checked before cold in
        # CLARIFYING_QUESTIONS dict order, so a message describing BOTH
        # used to ask only the fever age question and, once answered, only
        # ever recommend fever products — the cold side was silently
        # dropped from the entire conversation. See agent.py's
        # FEVER_AND_COLD_COMBINED_QUESTION / _resolve_fever_and_cold.
        "turns": [
            {
                "text": "i think i have running nose and high temperature",
                "checks": [
                    ("asks one combined child/adult question", lambda r: _contains_any(r, ["child", "adult"])),
                    ("no leaked tool name", _no_leaked_tool_name),
                ],
            },
            {
                "text": "for myself",
                "checks": [
                    ("recommends a real fever product", lambda r: "Paracetamol 500mg" in r),
                    ("also recommends a real cold product, not just fever", lambda r: "Cetirizine" in r),
                ],
            },
        ],
    },
    {
        "name": "bare_email_is_never_saved_as_the_address",
        # Real bug this pins: an email containing a digit
        # (sukumar123@example.com) has a digit and is long enough to pass
        # the address heuristic's basic check, and _extract_email then
        # stripped it out and left nothing behind — the caller's
        # `address_only or address_text` fallback re-saved the raw email
        # text as the shipping address, so orders confirmed "Shipping to:
        # sukumar123@example.com". See main.py's _is_bare_email.
        "turns": [
            {"text": "I have a dry cough", "checks": [("asks the clarifying question", lambda r: _contains_any(r, ["child", "adult"]))]},
            {"text": "for myself", "checks": [("recommends the real product", lambda r: "Cough Suppressant" in r)]},
            {"text": "yes", "checks": [("asks for a shipping address", lambda r: "shipping address" in r.lower())]},
            {
                "text": "sukumar123@example.com",
                "checks": [
                    ("never confirms an order to the bare email as the address", lambda r: "sukumar123@example.com" not in r),
                    ("never claims the order is placed", lambda r: "Order confirmed!" not in r),
                ],
            },
        ],
    },
    {
        "name": "bare_filler_phrase_is_never_saved_as_the_address",
        # Real bug this pins: "ship to" has no digit and is long enough to
        # pass the generic first-time-address fallback, and
        # ADDRESS_FILLER_RE's own \s+ after "to" never matches with
        # nothing following it — _strip_address_filler left the literal
        # filler phrase intact, so it got saved and shown back as the
        # shipping address verbatim. See main.py's _is_filler_only.
        "turns": [
            {"text": "I have a dry cough", "checks": [("asks the clarifying question", lambda r: _contains_any(r, ["child", "adult"]))]},
            {"text": "for myself", "checks": [("recommends the real product", lambda r: "Cough Suppressant" in r)]},
            {"text": "yes", "checks": [("asks for a shipping address", lambda r: "shipping address" in r.lower())]},
            {
                "text": "ship to",
                "checks": [
                    ("never confirms an order to the bare filler phrase as the address", lambda r: "Shipping to: ship to" not in r),
                    ("never claims the order is placed", lambda r: "Order confirmed!" not in r),
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
    for case_number, case in enumerate(CONVERSATION_CASES, start=1):
        # A fresh isolated DB per case, not just once for the whole run —
        # this is a single-demo-user app by design, so without this a
        # later case would inherit the previous case's saved address/
        # orders (real, correct behavior for one shared user across
        # sessions — just not what each case's scripted turns assume
        # starting from a blank slate).
        _use_isolated_db()
        # Short on purpose: ChatRequest caps session_id at 64 chars (see
        # schemas.py), and some case names alone are longer than that. The
        # case name is already printed on the next line for readability.
        session_id = f"conversation-eval-{case_number}"
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
            for check_name, check_fn in turn.get("state_checks", []):
                total += 1
                ok = bool(check_fn(session_id))
                print(f"    [{'PASS' if ok else 'FAIL'}] (state) {check_name}")
                passed += ok

    print(f"\n{passed}/{total} checks passed")
    return passed, total


if __name__ == "__main__":
    passed, total = run_eval()
    sys.exit(0 if passed == total else 1)
