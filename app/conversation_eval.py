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

from app import agent, config, main, store

ORDER_ID_RE = re.compile(r"ord-\d{4}")
CITATION_RE = re.compile(r"\(Source: [\w.-]+\.md")


def _disable_real_email() -> None:
    """This harness drives the real chat route, and a developer machine's .env
    can hold live SMTP credentials: on such a machine any scripted order that
    reaches the confirmation-email step would send a REAL message through that
    account (found Oct 16, when a new case's receipt request did exactly that
    to an @example.com address). tools.send_confirmation_email falls back to
    its mock mode when SMTP isn't configured, so switch it off for the run."""
    config.SMTP_CONFIGURED = False


def _use_isolated_db() -> None:
    _disable_real_email()
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
        "name": "price_questions_come_from_the_catalog",
        # Fully deterministic. Before Oct 15 "how much does <product> cost?"
        # got "the price is not stated in our knowledge base" and "price of
        # col-003" got the out-of-scope decline. A price question asked while
        # an order awaits its address must not disturb that order.
        "turns": [
            {"text": "how much does Ibuprofen 200mg Tablets cost?", "checks": [("states the catalog price", lambda r: "$5.29" in r)]},
            {"text": "what's the price of col-003", "checks": [("states the catalog price", lambda r: "$8.49" in r)]},
            {"text": "what is your cheapest cold medicine", "checks": [("names the cheapest cold product", lambda r: "$5.99" in r)]},
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {
                "text": "how much is Paracetamol 500mg Tablets?",
                "checks": [("answers the price", lambda r: "$4.99" in r)],
                "state_checks": [("no order placed by a price question", _order_count_is(0))],
            },
            {
                "text": "7 Price Street, Pune",
                "checks": [("the waiting order still completes", lambda r: "Order confirmed!" in r and "Ibuprofen" in r)],
                "state_checks": [("one real order", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "catalog_listing_then_order_by_name",
        # Fully deterministic. "show me all your products" used to get the
        # out-of-scope decline.
        "turns": [
            {
                "text": "show me all your products",
                "checks": [
                    ("lists a fever and a cold product with prices", lambda r: "Ibuprofen 200mg Tablets — $5.29" in r and "Guaifenesin Expectorant Syrup — $6.79" in r),
                    ("is not the out-of-scope decline", lambda r: "can't help with that here" not in r.lower()),
                ],
            },
            {"text": "I'd like to order Guaifenesin Expectorant Syrup", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {
                "text": "3 List Road, Pune",
                "checks": [("orders the named product", lambda r: "Order confirmed!" in r and "Guaifenesin" in r)],
                "state_checks": [("one real order", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "email_given_up_front_and_receipt_requests",
        # Fully deterministic (mail is forced to mock mode by _disable_real_email).
        # Before Oct 16 "my email is X" ahead of an order got a greeting and
        # was not saved, "what email do you have for me?" got "what symptom
        # is this for?", and "send me the receipt" reprinted the order.
        "turns": [
            {
                "text": "my email is sukumar@example.com",
                "checks": [("confirms the email was saved", lambda r: "saved sukumar@example.com" in r)],
                "state_checks": [("the email is really saved", lambda sid: store.get_email("demo_user") == "sukumar@example.com")],
            },
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {
                "text": "10 High St, Leeds",
                "checks": [("the order is confirmed and the email goes out without asking", lambda r: "Order confirmed!" in r and "confirmation email has been sent" in r)],
                "state_checks": [("one real order", _order_count_is(1))],
            },
            {
                "text": "what email do you have for me?",
                "checks": [("reads back the saved email", lambda r: "sukumar@example.com" in r and "symptom" not in r.lower())],
            },
            {
                "text": "send me the receipt",
                "checks": [("sends the confirmation for the real order", lambda r: "ord-0001" in r and "sukumar@example.com" in r)],
                "state_checks": [("no extra order created", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "emergencies_are_escalated_not_sold_to",
        # Fully deterministic. On Oct 17 an overdose got the fever child/adult
        # question, a 2-month-old's fever got a product recommendation, and
        # chest pain got the out-of-scope decline. An order that was waiting
        # for its address must still be waiting afterwards.
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {
                "text": "I took 10 paracetamol tablets",
                "checks": [
                    ("sends them to emergency care", lambda r: "emergency number" in r and "poison control" in r),
                    ("recommends nothing", lambda r: "order" not in r.lower().replace("emergency", "")),
                ],
                "state_checks": [("no order was placed", _order_count_is(0))],
            },
            {
                "text": "my baby is 2 months old and has a fever",
                "checks": [("tells them to see a doctor now", lambda r: "under 3 months" in r and "doctor" in r)],
            },
            {
                "text": "diagnose my chest pain",
                "checks": [("escalates", lambda r: "emergency medical care" in r)],
            },
            {
                "text": "45 Safe Street, Pune",
                "checks": [("the waiting order still completes afterwards", lambda r: "Order confirmed!" in r)],
                "state_checks": [("exactly one real order", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "hinglish_symptoms_are_understood",
        # Fully deterministic. "mujhe bukhar hai" / "mera bachha ko khansi hai"
        # were declined as out of scope before Oct 17.
        "turns": [
            {"text": "mujhe bukhar hai", "checks": [("asks child or adult", lambda r: "child, or for an adult" in r)]},
            {"text": "khud ke liye", "checks": [("recommends an adult fever product", lambda r: "Paracetamol 500mg Tablets" in r and "Children's" not in r)]},
        ],
    },
    {
        "name": "hinglish_childs_cough_asks_dry_or_wet",
        "turns": [
            {"text": "mera bachha ko khansi hai", "checks": [("asks dry or wet", lambda r: "dry" in r.lower() and "mucus" in r.lower())]},
            {"text": "sukhi khansi hai", "checks": [("recommends the children's product", lambda r: "Children's Cold & Cough Syrup" in r)]},
        ],
    },
    {
        "name": "stated_conditions_screen_the_recommendations",
        # Fully deterministic. On Oct 21 "I have a fever and stomach ulcer"
        # and "I'm on warfarin" were both offered Ibuprofen, and "high blood
        # pressure and a stuffy nose" was recommended Pseudoephedrine.
        "turns": [
            {"text": "I have a fever and stomach ulcer", "checks": [("asks child or adult", lambda r: "child, or for an adult" in r)]},
            {
                "text": "for myself",
                "checks": [
                    ("offers the paracetamol products", lambda r: "(fev-001)" in r and "(fev-002)" in r),
                    ("does not offer ibuprofen", lambda r: "(fev-003)" not in r.split("Left out")[0]),
                    ("says why it was left out", lambda r: "Left out because of what you told me" in r and "stomach ulcer" in r),
                ],
            },
            {
                "text": "order Ibuprofen 200mg Tablets",
                "checks": [("refuses to order it", lambda r: "I can't set up an order for Ibuprofen" in r)],
                "state_checks": [("no order exists", _order_count_is(0))],
            },
            {"text": "2", "checks": [("the safe product can still be picked by number", lambda r: "address" in r.lower())]},
        ],
    },
    {
        "name": "high_or_long_fever_adds_a_see_a_doctor_note",
        # Fully deterministic. "fever 104 for 5 days" used to get the ordinary
        # adult list with no mention of seeing a doctor.
        "turns": [
            {"text": "I have fever 104 for 5 days", "checks": [("asks child or adult", lambda r: "child, or for an adult" in r)]},
            {
                "text": "for myself",
                "checks": [
                    ("still offers the products", lambda r: "(fev-001)" in r),
                    ("tells them a doctor should check it", lambda r: "should be checked by a doctor soon" in r),
                ],
            },
        ],
    },
    {
        "name": "brands_we_do_not_stock",
        # Fully deterministic. "do you have Advil" / "I want Tylenol" used to
        # get the out-of-scope decline.
        "turns": [
            {
                "text": "do you have Advil",
                "checks": [
                    ("offers the same-ingredient product", lambda r: "Ibuprofen 200mg Tablets (fev-003)" in r),
                    ("says the brand is not carried", lambda r: "We don't carry Advil" in r),
                ],
            },
            {"text": "I want Tylenol", "checks": [("offers plain paracetamol products", lambda r: "(fev-001)" in r and "(col-003)" not in r)]},
            {"text": "any NyQuil?", "checks": [("honest not-stocked reply", lambda r: r.startswith("We don't carry"))]},
        ],
    },
    {
        "name": "combination_questions_use_the_product_warnings",
        # Fully deterministic. The model used to answer "yes, together" from
        # general knowledge; the knowledge base warns against doubling up on
        # paracetamol and says nothing about ibuprofen + paracetamol.
        "turns": [
            {
                "text": "can I take paracetamol and Cold & Flu Multi-Symptom Relief together",
                "checks": [("says no and quotes the warning", lambda r: r.startswith("No —") and "accidental overdose" in r)],
            },
            {
                "text": "can I take ibuprofen and paracetamol together",
                "checks": [("does not improvise an answer", lambda r: "doesn't say whether" in r and "pharmacist" in r)],
            },
        ],
    },
    {
        "name": "quantity_change_while_an_order_waits",
        # Fully deterministic. On Oct 23 "make it 2" was saved as the shipping
        # address: the order was placed "Shipping to: make it 2".
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {
                "text": "make it 2",
                "checks": [("changes the waiting order", lambda r: r.startswith("Got it — 2 x Ibuprofen 200mg Tablets") and "address" in r.lower())],
                "state_checks": [("nothing was placed", _order_count_is(0))],
            },
            {
                "text": "9 Quantity Lane, Pune",
                "checks": [
                    ("the order has the new quantity", lambda r: "Quantity: 2" in r),
                    ("ships to the real address", lambda r: "Shipping to: 9 Quantity Lane, Pune" in r),
                ],
                "state_checks": [
                    ("one real order", _order_count_is(1)),
                    ("the saved address is the real one", lambda sid: store.get_address("demo_user") == "9 Quantity Lane, Pune"),
                ],
            },
        ],
    },
    {
        "name": "one_more_of_the_same_after_an_order",
        # Fully deterministic. On Oct 23 "one more of the same" reached the
        # model, which asked "what symptom is this for?".
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {
                "text": "10 High St, Leeds",
                "checks": [("order placed", lambda r: "Order confirmed!" in r)],
                "state_checks": [("one real order", _order_count_is(1))],
            },
            {
                "text": "two more of the same",
                "checks": [("asks to confirm the address on file", _asks_to_confirm_address("10 High St, Leeds"))],
                "state_checks": [("no second order yet", _order_count_is(1)), ("awaiting confirmation", _awaiting_address_confirmation)],
            },
            {
                "text": "yes",
                "checks": [("the second order has the quantity asked for", lambda r: "Order confirmed!" in r and "Quantity: 2" in r and "ord-0002" in r)],
                "state_checks": [("two real orders", _order_count_is(2))],
            },
        ],
    },
    {
        "name": "switch_the_waiting_order_to_another_product",
        # Fully deterministic. On Oct 23 "actually give me paracetamol 500mg
        # instead" got the child-or-adult question and left the old order
        # pending; on Oct 24 "order ibuprofen 400mg" was saved as the address.
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {
                "text": "actually give me paracetamol 500mg instead",
                "checks": [("asks for an address again", lambda r: "address" in r.lower() and "child, or for an adult" not in r)],
                "state_checks": [("nothing placed", _order_count_is(0))],
            },
            {
                "text": "order ibuprofen 400mg",
                "checks": [("is not taken as the address", lambda r: "Order confirmed!" not in r)],
                "state_checks": [("still nothing placed", _order_count_is(0))],
            },
            {
                "text": "3 Switch Street, Pune",
                "checks": [("the switched product is what ships", lambda r: "Order confirmed!" in r and "Paracetamol 500mg Tablets" in r)],
                "state_checks": [("one real order", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "confirming_with_nothing_waiting",
        # Fully deterministic. On Oct 24 "place the order" with nothing in
        # progress reached the model, which invented a shipping-address flow.
        "turns": [
            {
                "text": "place the order",
                "checks": [("says there is nothing to confirm", lambda r: "nothing waiting for me to confirm" in r)],
                "state_checks": [("no order exists", _order_count_is(0))],
            },
            {
                "text": "order Ibuprofen 200mg Tablets",
                "checks": [("a real order flow starts when one is named", lambda r: "address" in r.lower())],
            },
            {
                "text": "5 Confirm Road, Pune",
                "checks": [("and completes", lambda r: "Order confirmed!" in r)],
                "state_checks": [("one real order", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "replies_to_the_receipt_email_prompt",
        # Fully deterministic (mail is forced to mock mode). Before Oct 25
        # "yes" got "Your address has been saved" and "later" got "You've saved
        # your address" -- neither happened.
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {"text": "10 High St, Leeds", "checks": [("order placed, email prompt shown", lambda r: "Order confirmed!" in r and "reply with your email address" in r)]},
            {
                "text": "yes please",
                "checks": [("asks which email, claims nothing was saved", lambda r: "what email address should I send the confirmation for order ord-0001" in r and "saved" not in r.lower())],
            },
            {
                "text": "a@b.com",
                "checks": [("sends it", lambda r: "sent the confirmation for order ord-0001 to a@b.com" in r)],
                "state_checks": [("email really saved", lambda sid: store.get_email("demo_user") == "a@b.com")],
            },
        ],
    },
    {
        "name": "skipping_the_receipt_email_prompt",
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {"text": "10 High St, Leeds", "checks": [("email prompt shown", lambda r: "reply with your email address" in r)]},
            {
                "text": "not now",
                "checks": [("closes the prompt without the out-of-scope decline", lambda r: "already confirmed without an email receipt" in r)],
                "state_checks": [("the order is untouched", _order_count_is(1))],
            },
        ],
    },
    {
        "name": "brand_names_in_combination_questions",
        # Fully deterministic. "Tylenol with Cold & Flu" is paracetamol twice;
        # aspirin and ibuprofen are both NSAIDs, which the knowledge base says
        # ibuprofen must not be combined with; Tylenol + Advil is a pair the
        # knowledge base is silent on.
        "turns": [
            {"text": "can I take Tylenol with Cold & Flu Multi-Symptom Relief", "checks": [("warns about doubling paracetamol", lambda r: r.startswith("No —") and "accidental overdose" in r)]},
            {"text": "can I take aspirin and ibuprofen together", "checks": [("warns about two NSAIDs", lambda r: r.startswith("No —") and "other NSAIDs" in r)]},
            {"text": "can I take Tylenol and Advil together", "checks": [("does not improvise", lambda r: "doesn't say whether" in r and "pharmacist" in r)]},
        ],
    },
    {
        "name": "who_it_is_for_is_taken_from_the_message",
        # Fully deterministic. On Oct 26 "my 3 year old has a fever" and "my
        # wife has a fever" were still asked "child or adult?", and "my 5 year
        # old has a dry cough" never reached the children's product.
        "turns": [
            {"text": "my 3 year old has a fever", "checks": [
                ("goes straight to the children's syrup", lambda r: "Children's Paracetamol Syrup" in r and "child, or for an adult" not in r),
                ("never offers the adult products", lambda r: "fev-001" not in r and "Ibuprofen" not in r)]},
            {"text": "my wife has a fever", "checks": [("goes straight to the adult list", lambda r: "(fev-001)" in r and "child, or for an adult" not in r)]},
            {"text": "my 5 year old has a dry cough", "checks": [("reaches the children's cough product", lambda r: "Children's Cold & Cough Syrup" in r)]},
            {"text": "my teenage son has a fever", "checks": [("a teenager is still asked", lambda r: "child, or for an adult" in r)]},
        ],
    },
    {
        "name": "cancel_last_and_cancel_all",
        # Fully deterministic. On Oct 27 "cancel my last order" got "I don't
        # have any information about an order with that ID", and "cancel all my
        # orders" got the out-of-scope decline.
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {"text": "10 High St, Leeds", "checks": [("first order placed", lambda r: "ord-0001" in r)], "state_checks": [("one order", _order_count_is(1))]},
            {"text": "order Cetirizine 10mg Antihistamine Tablets", "checks": [("asks to confirm the address", _asks_to_confirm_address("10 High St, Leeds"))]},
            {"text": "yes", "checks": [("second order placed", lambda r: "ord-0002" in r)], "state_checks": [("two orders", _order_count_is(2))]},
            {
                "text": "cancel my last order",
                "checks": [("cancels only the newest", lambda r: "ord-0002" in r and "has been cancelled" in r and "ord-0001" not in r)],
                "state_checks": [("the first is untouched", _order_status_is("ord-0001", "placed")), ("the last is cancelled", _order_status_is("ord-0002", "cancelled"))],
            },
            {
                "text": "cancel all my orders",
                "checks": [("cancels the one still active", lambda r: "ord-0001" in r and "has been cancelled" in r)],
                "state_checks": [("everything is cancelled", _order_status_is("ord-0001", "cancelled"))],
            },
        ],
    },
    {
        "name": "shop_policy_questions_are_answered_honestly",
        # Fully deterministic. There is no policy data, so any specific answer
        # about delivery, refunds or payment would be invented; before Oct 27
        # these got the out-of-scope decline, the order list, or the catalog.
        "turns": [
            {"text": "order Ibuprofen 200mg Tablets", "checks": [("asks for an address", lambda r: "address" in r.lower())]},
            {"text": "10 High St, Leeds", "checks": [("order placed", lambda r: "ord-0001" in r)]},
            {"text": "how long does delivery take", "checks": [
                ("says it does not know", lambda r: "I don't have information about delivery times or tracking" in r),
                ("still shows the real order status", lambda r: "Order ID: ord-0001" in r)]},
            {"text": "what is your refund policy", "checks": [("says it does not know", lambda r: "I don't have information about refunds or returns" in r and "Order ID" not in r)]},
            {"text": "can I return the medicine", "checks": [("does not list the catalog", lambda r: "Here's what we carry" not in r)]},
            {"text": "can I change my delivery address for ord-0001", "checks": [("explains what is possible", lambda r: "can't change the address on an order that's already placed" in r)],
             "state_checks": [("the order is untouched", _order_status_is("ord-0001", "placed"))]},
        ],
    },
    {
        "name": "cough_with_fever_and_cold_symptoms_is_planned_as_a_whole",
        # Fully deterministic. Before Oct 28 "fever and a dry cough" got the
        # cough product only, and after "dry or chesty?" the reply "for myself"
        # was declined as out of scope.
        "turns": [
            {"text": "I'm an adult with a fever and a dry cough", "checks": [
                ("recommends the fever products", lambda r: "For the fever:" in r and "(fev-001)" in r),
                ("and the dry-cough product", lambda r: "For the dry cough:" in r and "(col-004)" in r)]},
            {"text": "I have a fever, cough, and runny nose", "checks": [("asks what is missing in one question", lambda r: r.startswith("Two quick questions"))]},
            {"text": "for myself", "checks": [("asks only about the cough now", lambda r: "dry" in r.lower() and "Two quick" not in r and "can't help with that" not in r)]},
            {"text": "chesty", "checks": [
                ("all three symptom groups", lambda r: "For the fever:" in r and "chesty (productive) cough" in r and "For the cold symptoms" in r),
                ("the paracetamol combination product is left out", lambda r: "(col-003)" not in r)]},
            {"text": "my 4 year old has a fever, dry cough and runny nose", "checks": [("only children's products", lambda r: "(fev-004)" in r and "(col-006)" in r and "(fev-001)" not in r)]},
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
        "name": "dose_answers_contain_only_documented_figures",
        # Needs the real LLM. On Oct 25 the model answered "maximum dose of
        # ibuprofen" with the documented 6 tablets / 1200mg PLUS an invented
        # "3-4 tablets (600-800mg) every 4-6 hours" (the knowledge base says 1-2
        # tablets / 200-400mg per dose). Whatever the model writes, a figure the
        # knowledge base does not state must never reach the customer.
        "turns": [
            {
                "text": "what is the maximum dose of ibuprofen per day",
                "checks": [
                    ("states the documented daily maximum", lambda r: "1200" in r),
                    ("contains no invented single dose", lambda r: "600-800" not in r and "3-4 tablets" not in r),
                    ("cites a real knowledge-base source", lambda r: CITATION_RE.search(r) is not None),
                ],
            },
        ],
    },
    {
        "name": "a_recommendation_never_names_a_medicine_we_do_not_carry",
        # Needs the real LLM. On Oct 22-26 the model's free text recommended
        # "Acetaminophen (Tylenol)", "ibuprofen (Advil, Motrin)" and
        # "Benadryl". Whatever it writes, a recommendation of a brand we don't
        # stock must come out as what we do carry. The questions below avoid
        # naming the brand themselves, so any brand in the reply is the model's.
        "turns": [
            {
                "text": "what can I take for a fever if I am allergic to paracetamol",
                "checks": [
                    ("no brand we don't carry is recommended", lambda r: not re.search(r"\b(?:take|try|use|consider|suggest|recommend)\w*\b[^.\n]{0,80}\b(?:tylenol|advil|motrin|nurofen|panadol|crocin|dolo|benadryl|aspirin)\b", r, re.IGNORECASE)),
                ],
            },
            {
                "text": "I have a cough and a runny nose, I am an adult, anything besides your usual list",
                "checks": [
                    ("no brand we don't carry is recommended", lambda r: not re.search(r"\b(?:take|try|use|consider|suggest|recommend)\w*\b[^.\n]{0,80}\b(?:nyquil|dayquil|benadryl|robitussin|mucinex|sudafed|zyrtec|vicks)\b", r, re.IGNORECASE)),
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
