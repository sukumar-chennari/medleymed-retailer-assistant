import json
import logging
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app import guardrails, store, tools
from app.agent import (
    INFO_QUESTION_RE,
    ORDER_INTENT_RE,
    _has_reorder_intent,
    _has_unnegated_intent,
    resolve_clarification,
    run_turn,
)
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest

logger = logging.getLogger("medicine_assistant")

app = FastAPI()

GENERIC_FAILURE_REPLY = (
    "Sorry, I ran into a problem processing that. Please try again — if it "
    "keeps happening, check that Ollama is running."
)

# The domain is dot-separated labels that each END in a word character. Real
# bug: the old tail `[\w.-]+` also matched trailing punctuation, so
# "my email is bob@gmail.com." captured "bob@gmail.com." — a trailing-dot
# address saved as the customer's email and used as the confirmation
# recipient, which a mail server rejects.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]*\w)+")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard():
    user = store.get_user("demo_user")
    orders = store.list_orders("demo_user")
    return {
        "name": user["name"],
        "email": user["email"],
        "address": user["address"],
        "orders": orders,
        "catalog_count": len(store.get_catalog()),
    }


@app.get("/api/metrics")
def metrics():
    return store.get_metrics_summary()


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    store.log_metric_event(req.session_id, "feedback", req.rating, detail=req.reply_snippet[:200] or None)
    return {"ok": True}


@app.get("/api/catalog")
def catalog():
    return store.get_catalog()


@app.post("/api/reset-demo")
def reset_demo():
    """Wipes orders, session state, and metrics, and resets the saved
    address/email — lets a live demo be re-run from a clean slate without
    restarting the server or touching the database file by hand."""
    store.reset_demo_data()
    return {"ok": True}


def _extract_email(text: str) -> tuple[str | None, str]:
    """Pulls an email address out of free text if present, e.g. a user typing
    their address and email in the same message. Returns (email_or_None,
    remaining_text_with_email_removed)."""
    match = EMAIL_RE.search(text)
    if not match:
        return None, text
    email = match.group(0)
    remainder = text[: match.start()] + text[match.end() :]
    # Also drop a leading label word like "email:" or "mail" that introduced
    # the address, so it doesn't linger in the saved shipping address.
    remainder = re.sub(r"[,;:\-\s]*\b(e-?mail|mail)\b\s*:?\s*$", "", remainder, flags=re.IGNORECASE)
    # Punctuation around the email is left behind: sentence punctuation
    # ("...bob@gmail.com.") now that the pattern no longer swallows it, and
    # the wrapper in "(bob@gmail.com)", "<bob@gmail.com>" or a quoted one,
    # which already left "()"/"<>" behind. Without stripping it a bare email
    # isn't recognized as bare (see _is_bare_email) and the stray
    # punctuation is saved as the shipping address.
    remainder = remainder.strip(" ,;:-.!?()<>[]\"'")
    return email, remainder


def _looks_like_an_email(text: str) -> bool:
    return EMAIL_RE.search(text) is not None


def _is_bare_email(text: str) -> bool:
    """True when the text is (almost) entirely just an email address, with
    no actual address content alongside it — e.g. answering "what's your
    shipping address?" with only an email. Real bug this guards against:
    _extract_email would strip the email and leave nothing behind, so
    _complete_pending_order's `address_only or address_text` fallback used
    to re-add the raw text — email included — back in as the "address",
    saving and later shipping to the email itself."""
    if not _looks_like_an_email(text):
        return False
    _, remainder = _extract_email(text)
    return not remainder.strip()


ADDRESS_FILLER_RE = re.compile(
    r"^\s*(actually\s+)?(please\s+)?(ship|send|deliver)\s+(it\s+|this\s+|that\s+)?to\s+",
    re.IGNORECASE,
)


def _strip_address_filler(text: str) -> str:
    """Naturally answering "should I ship to X, or give a different one?"
    tends to produce phrasing like "actually ship to 456 New Ave" rather
    than just the bare address — without this, that whole phrase gets saved
    and shown back verbatim in the order confirmation."""
    return ADDRESS_FILLER_RE.sub("", text).strip()


_ADDRESS_FILLER_ONLY_RE = re.compile(
    r"^\s*(actually\s+)?(please\s+)?(ship|send|deliver)\s+(it\s+|this\s+|that\s+)?to\s*$",
    re.IGNORECASE,
)


def _is_filler_only(text: str) -> bool:
    """True when the text is ONLY the filler phrase ADDRESS_FILLER_RE
    exists to strip, with nothing substantive after it — e.g. a message
    cut short mid-composition. Real bug this guards against:
    _looks_like_a_first_time_address's generic fallback (no digit
    required) accepted bare filler phrases like "ship to" or "please ship
    it to" as a real address, since ADDRESS_FILLER_RE's own \\s+ after
    "to" never matches with nothing following — _strip_address_filler
    left the literal filler phrase intact, and it got saved and shown
    back as the shipping address verbatim."""
    return bool(_ADDRESS_FILLER_ONLY_RE.match(text))


_DRUG_WITH_STRENGTH_RE = re.compile(r"^[a-z]+ \d+\s?mg\b")


def _names_a_catalog_product(text: str) -> bool:
    """True when the text contains a catalog product's name, or its drug name
    with the strength ("ibuprofen 200mg"). Such text is a product reference,
    never a shipping address — but the name's digits ("200mg") and length
    used to be enough for _looks_like_an_address to accept it, so a pending
    order took "I'd like to order Ibuprofen 200mg Tablets" as the address
    and shipped the FIRST product to that sentence."""
    normalized = " ".join(text.lower().replace("\u2019", "'").split())
    for product in store.get_catalog():
        name = product["name"].lower()
        if name in normalized:
            return True
        short = _DRUG_WITH_STRENGTH_RE.match(name)
        if short and short.group(0) in normalized:
            return True
    return False


def _looks_like_an_address(text: str) -> bool:
    """A pending order treats the user's next message as their shipping address —
    but only if it plausibly is one. Without this, a reply like "thank you" or
    "actually never mind" gets silently saved and used as the shipping address.
    Also rejects anything containing a product id pattern (fev-001, col-002) —
    a message like "order col-001" has a digit and is long enough to pass the
    basic check, but is clearly a product reference, not an address."""
    if guardrails.PRODUCT_ID_RE.search(text):
        return False
    if _names_a_catalog_product(text) or _names_an_order_id(text):
        return False
    if _is_bare_email(text):
        return False
    return any(ch.isdigit() for ch in text) and len(text) >= 8


# A bare place name ("hyderabad") is a completely normal way to answer a
# fresh "what's your address?" ask, but has no digit — without this, it
# fell through to the general LLM turn, which is exactly what let the model
# claim "I've saved your address" ungrounded (see
# guardrails.claims_address_saved for the actual safety net this depends on
# regardless). Deliberately scoped to ONLY the pending_product_id
# first-time-address context (see chat()) rather than folded into
# _looks_like_an_address itself — the pending_address_confirmation context
# has a real third option ("sure, whatever you think is best" meaning
# "just use what's on file") that this heuristic would wrongly capture as
# a brand new address if applied there too.
_QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how", "is", "are", "can",
    "could", "would", "will", "do", "does", "did", "should",
}


def _looks_like_a_first_time_address(text: str) -> bool:
    if _looks_like_an_address(text):
        return True
    if guardrails.PRODUCT_ID_RE.search(text):
        return False
    if _names_a_catalog_product(text) or _names_an_order_id(text):
        return False
    if _is_bare_email(text):
        return False
    if _is_filler_only(text):
        return False
    stripped = text.strip()
    if not (2 <= len(stripped) <= 80):
        return False
    if "?" in stripped:
        return False
    first_word = re.split(r"[\s,]+", stripped.lower(), maxsplit=1)[0]
    if first_word in _QUESTION_WORDS:
        return False
    if _declines(stripped):
        return False
    if tools.classify(stripped) is not None:
        return False
    return True


def _complete_pending_order(session_id: str, pending: dict, address_text: str) -> str:
    """Deterministically finishes an order once the user's next message supplies
    the address, instead of trusting the LLM to chain save_address + place_order
    + send_confirmation_email itself across turns (see agent.py/tools.py for why)."""
    email_in_text, address_only = _extract_email(address_text)
    store.save_address("demo_user", _strip_address_filler(address_only or address_text))
    if email_in_text:
        store.save_email("demo_user", email_in_text)

    order = json.loads(tools.place_order(pending["product_id"], pending.get("quantity", 1)))
    if "error" in order:
        return f"Sorry, something went wrong placing that order: {order['error']}"

    recipient_email = store.get_email("demo_user")
    summary = (
        f"Order {order['order_id']}: {order['quantity']}x {order['product_name']} "
        f"(${order['total_price_usd']}) to {order['address']}"
    )

    if recipient_email:
        email_result = json.loads(tools.send_confirmation_email(recipient_email, summary))
        order["email_sent"] = email_result.get("sent", False)
    else:
        # No email on file at all — the order still ships fine (address is
        # all that requires), but ask for an email so a confirmation can go
        # out; the next message is captured as the email, same pattern as
        # the address flow above.
        store.set_pending_email(session_id, order["order_id"])
        order["email_sent"] = False

    return guardrails.build_order_confirmation(order)


def _complete_confirmed_order(session_id: str, pending: dict) -> str:
    """Finishes an order once the user has confirmed the address already on
    file is still correct, rather than assuming it silently — real orders
    always ship somewhere real; a stale saved address is a genuine mistake
    worth double-checking, not just a demo nicety."""
    order = json.loads(tools.complete_confirmed_order(session_id, pending["product_id"], pending.get("quantity", 1)))
    if "error" in order:
        return f"Sorry, something went wrong placing that order: {order['error']}"
    return guardrails.build_order_confirmation(order)


# Filler words stripped before checking for an ordinal/positional selection
# phrase — "last item bro" and "the second one please" both need to reduce
# to just the meaningful word(s) before matching.
_SELECTION_FILLER_WORDS = {
    "bro", "please", "pls", "the", "that", "item", "items", "option",
    "options", "product", "products", "number", "no",
}
_ORDINAL_TO_INDEX = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
}


def _resolve_bare_selection(text: str, last_products: list[dict] | None) -> dict | None:
    """After a product list is shown, users very naturally reply with just a
    number ("3"), an ordinal phrase ("last one", "the third item bro"), or an
    id fragment ("001") rather than the full product_id — and the model
    handles bare numbers fine but ordinal phrases unreliably (observed: asked
    "last one" after a fever list, it fabricated an entirely unrelated cold
    product list instead of picking the actual last fever item — presumably
    a hallucinated fresh lookup_symptom call with nothing grounding it).
    Resolving these deterministically against whatever list was actually
    just shown removes that failure mode the same way bare digits already
    are handled — matching by the product id's numeric suffix first for
    plain digits (unambiguous even though ids like fev-001/col-001 collide
    numerically across categories, since we only ever look inside the
    specific list just shown), falling back to plain list position; ordinal
    phrases resolve to list position directly, since "last"/"third" refer to
    what was actually displayed, not a catalog id."""
    if not last_products:
        return None
    t = text.strip().lower().strip("!.,? ")

    if re.fullmatch(r"\d{1,3}", t):
        n = int(t)
        for product in last_products:
            suffix = product["id"].split("-")[-1]
            if suffix.isdigit() and int(suffix) == n:
                return product
        if 1 <= n <= len(last_products):
            return last_products[n - 1]
        return None

    words = [w for w in re.findall(r"[a-z0-9]+", t) if w not in _SELECTION_FILLER_WORDS]
    if not words:
        return None

    # "one" doubles as a generic filler noun when paired with an ordinal
    # word ("second one", "last one") but means position 1 on its own
    # ("one" / "just one").
    if len(words) > 1 and "one" in words and (words[0] == "last" or words[0] in _ORDINAL_TO_INDEX):
        words = [w for w in words if w != "one"]

    def _at(n: int) -> dict | None:
        return last_products[n - 1] if 1 <= n <= len(last_products) else None

    if words == ["last"]:
        return last_products[-1]
    if words == ["one"]:
        return _at(1)
    if len(words) == 1 and words[0] in _ORDINAL_TO_INDEX:
        return _at(_ORDINAL_TO_INDEX[words[0]])
    return None


# Every word a purely-affirmative reply could plausibly be built from.
# "yes please" broke the old exact-phrase-list version of this check (it
# wasn't one of the enumerated phrases, so it fell through to the LLM, which
# then called start_order with a product *name* instead of its id and
# produced a confused "I made an error" reply). Checking that EVERY word in
# the message belongs to this set (rather than ANY word) fixes that without
# reopening the exact risk the original design avoided: a message like "ok
# but what about the ibuprofen instead" has words ("but", "ibuprofen",
# "instead") outside this set, so it still correctly fails the check.
AFFIRMATIVE_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "confirm",
    "go", "ahead", "do", "it", "place", "order", "the", "sounds", "good",
    "great", "perfect", "thats", "works", "lets",
    # Casual variants observed in real testing — "ok continue bro" fell
    # through to the generic re-ask because "continue" and "bro" weren't
    # recognized, producing a stuck loop even though the intent was clearly
    # to proceed.
    "continue", "proceed", "bro", "dude", "man", "correct", "right", "fine",
    "cool", "alright",
}


def _is_affirmative(text: str) -> bool:
    """A bare "yes"/"ok"/"yes please" is the natural way to respond to
    "would you like to order this?"."""
    normalized = text.strip().lower().strip("!.,? ")
    words = re.findall(r"[a-z']+", normalized)
    return bool(words) and all(w in AFFIRMATIVE_WORDS for w in words)


# Signals like "no new one" / "i'll give another address" / "different one"
# reject the address on file but don't yet supply a replacement — distinct
# from genuine ambiguity. Observed failure: several such phrasings in a row
# ("no new one", "i will give another address", "i will give different
# one") all hit the same unhelpful "should I ship to X, or give a different
# address?" re-ask verbatim, producing a real stuck loop with no progress.
ADDRESS_REJECTION_WORDS = {"no", "not", "different", "another", "new", "change", "elsewhere", "else", "other"}
# Real bug: 6 was picked by only checking the shortest known-good
# phrasings in tests/test_main_heuristics.py, without also checking
# conversation_eval.py's own established "no, i will give a different
# one" (7 words) — which this cap then silently broke, regressing a
# previously-passing scripted conversation. 8 covers that phrase with a
# little headroom while still well under the 9-word false-positive case
# ("actually, do you have any other cold medicine options") this cap
# exists to reject.
MAX_ADDRESS_REJECTION_WORDS = 8


def _wants_different_address(text: str) -> bool:
    # Real bug: unlike _declines below (which has this same cap for the
    # same reason), this had no length limit at all — several of
    # ADDRESS_REJECTION_WORDS ("no", "new", "other"...) are common enough
    # that an unrelated, longer message asking a real question ("do you
    # have any other cold medicine options") got misread as rejecting the
    # address on file, derailing the conversation into "what's the new
    # shipping address?" instead of ever answering the actual question.
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > MAX_ADDRESS_REJECTION_WORDS:
        return False
    return any(w in ADDRESS_REJECTION_WORDS for w in words)


# Short + contains a clear decline word — same conservative pattern as
# guardrails.py's GREETING_WORDS/BYE_WORDS (a short message containing one
# of these is basically never anything else). Used to detect the user
# opting out of a pending email/address ask rather than answering it.
DECLINE_WORDS = {"no", "nope", "nah", "skip", "cancel", "nevermind"}
MAX_DECLINE_WORDS = 6


def _declines(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > MAX_DECLINE_WORDS:
        return False
    return any(w in DECLINE_WORDS for w in words)


def _looks_like_order_cancellation(text: str) -> bool:
    """"cancel" is in DECLINE_WORDS because "cancel that"/"cancel" alone is a
    natural way to opt out of a pending email/address ask — but "cancel my
    order" right after an order was just placed (pending_email_order_id
    context) is almost never that; it's a request to actually cancel the
    real order, which needs the agent's real cancel_order tool. Found via
    conversation_eval.py: "please cancel my order" was silently swallowed
    as "no, skip the confirmation email" and the order was never touched."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return "cancel" in words and ("order" in words or _names_an_order_id(text))


_ORDER_ID_IN_TEXT_RE = re.compile(r"\bord-\d+\b", re.IGNORECASE)


def _names_an_order_id(text: str) -> bool:
    """A message naming a specific order ("cancel ord-0002 please") is about
    that order, never an answer to a pending email/address question. Real
    bug: with an email prompt outstanding, that message matched the decline
    words ("cancel") and got "the order's already confirmed without an email
    receipt" — the order was not cancelled and the customer was told it was
    fine. The same words are consumed by the two address-pending decline
    branches, which also told the customer nothing was placed."""
    return bool(_ORDER_ID_IN_TEXT_RE.search(text))


# A plain "reorder that" must be handled in code, not left to the model.
# Real bug, seen in 3 of ~8 live conversation_eval.py runs: the model skipped
# its reorder_last tool call and either invented a receipt for an order that
# never existed (now caught by the "order id:" guard) or fell back to a
# generic reply — so no address confirmation was ever set up, and the next
# message ("no, I'll give a different one") fell through to the model too.
# Same "don't trust the model with a state change" treatment as a bare
# "yes" or a numbered selection. Deliberately only the PLAIN request: any
# quantity ("2 of them", "another", "twice"), product id, email, or a long
# message is left to the model, which can use reorder_last's quantity
# argument.
_QUANTITY_WORDS_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twice|double|couple|pair|another|more|extra)\b",
    re.IGNORECASE,
)
MAX_REORDER_REQUEST_WORDS = 8


def _is_plain_reorder_request(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > MAX_REORDER_REQUEST_WORDS:
        return False
    if "@" in text or guardrails.PRODUCT_ID_RE.search(text) or _QUANTITY_WORDS_RE.search(text):
        return False
    return _has_reorder_intent(text)


# A message that clearly orders ONE specific catalog product by name — which is
# exactly what the dashboard's "Ask Assistant" button sends ("I'd like to
# order <product name>") — must start that order, not be put through the
# symptom-clarification machinery. Real bug, found by sending that button's
# text for all 10 catalog products through the real route: 0 of 10 started an
# order. Most got "Is this fever for a child, or for an adult/yourself?" —
# including "Children's Paracetamol Syrup" — and "Cough Suppressant Syrup"
# got "is your cough dry or wet?"; after answering, the customer was told the
# product they had already named "would be a good fit" and STILL hadn't been
# asked for an address; and the model told one customer it "couldn't find
# Guaifenesin Expectorant Syrup in our catalog". Handled in code, like a bare
# numbered selection and a plain "reorder that". Only the clear case:
# unnegated order intent, exactly one product named (by full name or id), not
# an information question, and at most one unambiguous quantity cue once the
# name is removed (the names themselves contain numbers like "500mg") —
# anything else is left to the model. A quantity ("order 2 Ibuprofen 200mg
# Tablets", "... twice") was first left to the model too, and it re-asked the
# age question or invented a product ("Col-004, which is a fever medicine"):
# 0 of 5 phrasings reached a pending order.
_MAX_NAMED_ORDER_WORDS = 14

# Quantity cues a named order may carry. "another/more/extra" are NOT here:
# they depend on what was ordered before, so the model gets those.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twice": 2, "double": 2,
    "couple": 2, "pair": 2,
}
_QUANTITY_TOKEN_RE = re.compile(r"\b(?:\d+|" + "|".join(_NUMBER_WORDS) + r")\b")
_AMBIGUOUS_QUANTITY_RE = re.compile(r"\b(?:another|more|extra)\b")


def _parse_quantity(rest: str) -> int | None:
    """The one explicit quantity in `rest` (the message with the product name
    removed), 1 when there is none, None when it is ambiguous (two cues, or
    "another"/"more"/"extra"). Out-of-range values are returned as-is:
    tools.start_order clamps them and says so."""
    if _AMBIGUOUS_QUANTITY_RE.search(rest):
        return None
    tokens = _QUANTITY_TOKEN_RE.findall(re.sub(r"\bx(\d+)\b", r"\1", rest))
    if not tokens:
        return 1
    if len(tokens) > 1:
        return None
    token = tokens[0]
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


def _resolve_named_product_order(text: str) -> tuple[dict, int] | None:
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > _MAX_NAMED_ORDER_WORDS or "@" in text:
        return None
    if words[0] in _QUESTION_WORDS or INFO_QUESTION_RE.search(text):
        return None
    if not _has_unnegated_intent(text, ORDER_INTENT_RE):
        return None

    normalized = " ".join(text.lower().replace("\u2019", "'").split())
    named = {}
    for product in store.get_catalog():
        name = product["name"].lower()
        if name in normalized:
            named[product["id"]] = (product, name)
    for product_id in {m.lower() for m in guardrails.PRODUCT_ID_RE.findall(text)}:
        product = store.find_product(product_id)
        if product:
            named.setdefault(product["id"], (product, product["id"]))
    if len(named) != 1:
        return None
    product, matched = next(iter(named.values()))

    quantity = _parse_quantity(normalized.replace(matched, " "))
    if quantity is None:
        return None
    return product, quantity


def _reply_for_start_order_result(order: dict) -> str:
    if "error" in order:
        return f"Sorry, {order['error']}"

    if not order.get("order_placed"):
        # tools.start_order already recorded the pending order/confirmation
        # internally; we just need to ask the user ourselves.
        return guardrails.reply_for_deferred_order(order)

    return guardrails.build_order_confirmation(order)


def _complete_pending_email(email: str, order_id: str) -> str:
    store.save_email("demo_user", email)
    order = store.get_order(order_id)
    if order is None:
        return f"Thanks, I've saved {email} for next time — though I couldn't find that earlier order to send a confirmation for."

    # Real bug: nothing clears pending_email when an order is cancelled
    # while its confirmation email is still outstanding (see "please cancel
    # my order" right after placing one with no email on file yet) — without
    # this check, supplying the email afterward sent a cheerful "I've sent
    # the confirmation" for an order that no longer exists.
    if order["status"] == "cancelled":
        return f"That order was cancelled, so there's nothing to confirm — I've still saved {email} for next time."

    summary = f"Order {order['order_id']}: {order['product_name']} (${order['price_usd']}) to {order['address']}"
    email_result = json.loads(tools.send_confirmation_email(email, summary))
    if email_result.get("sent"):
        return f"Thanks! I've sent the confirmation for order {order['order_id']} to {email}."
    return f"I've saved {email}, but the confirmation email couldn't be sent right now."


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Real bug: this checked the raw req.text, so a whitespace-only message
    # ("   ") passed the check and went to the LLM as a blank user turn (which
    # then hallucinated a symptom lookup out of nothing). Strip first.
    if not (req.text and req.text.strip()) and not req.image_b64:
        raise HTTPException(status_code=400, detail="Provide text or an image.")

    messages = store.get_session_messages(req.session_id)
    text = req.text.strip() if req.text else ""

    try:
        pending_email_order_id = store.get_pending_email(req.session_id)
        pending_product_id = store.get_pending_order(req.session_id)
        pending_clarification = store.get_pending_clarification(req.session_id)
        pending_address_confirmation = store.get_pending_address_confirmation(req.session_id)
        last_products = store.get_last_products(req.session_id)
        last_recommended_product_id = store.get_last_recommended_product(req.session_id)
        selected_product = _resolve_bare_selection(text, last_products) if text else None

        if pending_clarification and text:
            # Same reasoning as the deterministic question itself (see
            # agent.run_turn): the model fabricated an answer to this
            # question once already rather than admitting it needed to ask,
            # so the answer doesn't get left to it either.
            store.clear_pending_clarification(req.session_id)
            resolved_reply = resolve_clarification(pending_clarification, text, req.session_id)
            if resolved_reply:
                reply = resolved_reply
                messages.append({"role": "user", "content": text})
                messages.append({"role": "assistant", "content": resolved_reply})
            else:
                reply, messages = run_turn(
                    messages,
                    user_text=req.text,
                    session_id=req.session_id,
                    image_b64=req.image_b64,
                    image_media_type=req.image_media_type,
                )
        elif pending_address_confirmation and text and _is_affirmative(text):
            store.clear_pending_address_confirmation(req.session_id)
            reply = _complete_confirmed_order(req.session_id, pending_address_confirmation)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_address_confirmation and text and _looks_like_an_address(text):
            # They gave a different address instead of confirming the one on
            # file — use the new one, same as the no-address-yet flow.
            store.clear_pending_address_confirmation(req.session_id)
            reply = _complete_pending_order(req.session_id, pending_address_confirmation, text)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_address_confirmation and text and _wants_different_address(text) and not _names_an_order_id(text):
            # They've signaled they want to give a new one but haven't
            # supplied it yet — ask specifically for it instead of repeating
            # the same compound question. pending_address_confirmation stays
            # set, so their very next message (the actual address) is still
            # handled by the _looks_like_an_address branch above.
            reply = "No problem — what's the new shipping address?"
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_address_confirmation and text and _declines(text) and not _names_an_order_id(text):
            store.clear_pending_address_confirmation(req.session_id)
            reply = "No problem — I won't place that order. Let me know if you change your mind."
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_address_confirmation and text:
            # Real bug, same trapping pattern already fixed for
            # pending_email_order_id/pending_product_id below but missing
            # here: this used to unconditionally repeat "Just to confirm —
            # should I ship to X..." even for a completely unrelated message
            # ("who is narendra modi"), trapping the conversation instead of
            # just answering it. Route it through the real turn instead —
            # while leaving pending_address_confirmation set, so a genuine
            # yes/address/rejection typed later is still handled by the
            # branches above.
            reply, messages = run_turn(
                messages,
                user_text=req.text,
                session_id=req.session_id,
                image_b64=req.image_b64,
                image_media_type=req.image_media_type,
            )
        elif text and not pending_product_id and _is_plain_reorder_request(text):
            # Placed before the pending-email branches on purpose: the usual
            # moment to say "reorder that" is right after an order was placed
            # with no email on file yet, i.e. with pending_email set, which
            # used to send this message to the model. pending_email stays set
            # (the customer can still give an email), same as before.
            order = json.loads(tools.reorder_last(req.session_id))
            reply = _reply_for_start_order_result(order)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif text and (named_order := _resolve_named_product_order(text)):
            # Next to the plain-reorder branch above, ahead of the pending-email
            # branches, for the same reason: this is usually said right after an
            # order was placed with no email on file yet (pending_email set).
            named_product, named_quantity = named_order
            order = json.loads(tools.start_order(named_product["id"], req.session_id, named_quantity))
            reply = _reply_for_start_order_result(order)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_email_order_id and text and _looks_like_an_email(text):
            store.clear_pending_email(req.session_id)
            email_in_text, _ = _extract_email(text)
            reply = _complete_pending_email(email_in_text, pending_email_order_id)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_product_id and text and _looks_like_a_first_time_address(text):
            store.clear_pending_order(req.session_id)
            reply = _complete_pending_order(req.session_id, pending_product_id, text)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_email_order_id and text and _declines(text) and not _looks_like_order_cancellation(text):
            store.clear_pending_email(req.session_id)
            reply = "No problem — the order's already confirmed without an email receipt. Anything else I can help with?"
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_email_order_id and text:
            # Doesn't look like an email or a decline — observed failure:
            # this used to unconditionally repeat "I still need an email
            # address..." even for a completely unrelated message ("who is
            # narendra modi"), trapping the conversation instead of just
            # answering it. The order itself is already placed regardless
            # of email, so there's nothing risky about handling the actual
            # message normally — route it through the real turn (still
            # correctly declines out-of-scope, or continues normally) while
            # leaving pending_email_order_id set, so a genuine email typed
            # later is still captured by the branch above.
            reply, messages = run_turn(
                messages,
                user_text=req.text,
                session_id=req.session_id,
                image_b64=req.image_b64,
                image_media_type=req.image_media_type,
            )
        elif pending_product_id and text and _declines(text) and not _names_an_order_id(text):
            store.clear_pending_order(req.session_id)
            reply = "No problem — I won't place that order without a shipping address. Let me know if you change your mind."
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_product_id and text:
            # Reply doesn't look like an address, a bare "yes" (fabricated a
            # plausible-looking address once — see the address-confirmation
            # guard elsewhere), or a decline. Observed failure: a message
            # asking something else entirely ("who is narendra modi") still
            # got "I still need your shipping address..." instead of an
            # actual answer, trapping the conversation. Route it through the
            # real turn instead — it still correctly declines out-of-scope
            # or continues normally — while leaving pending_product_id set,
            # so a real address typed later still completes this order via
            # the branch above.
            reply, messages = run_turn(
                messages,
                user_text=req.text,
                session_id=req.session_id,
                image_b64=req.image_b64,
                image_media_type=req.image_media_type,
            )
        elif selected_product and not pending_product_id:
            order = json.loads(tools.start_order(selected_product["id"], req.session_id))
            reply = _reply_for_start_order_result(order)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif (
            last_recommended_product_id
            and text
            and _is_affirmative(text)
            and not pending_product_id
            and not pending_email_order_id
        ):
            store.clear_last_recommended_product(req.session_id)
            order = json.loads(tools.start_order(last_recommended_product_id, req.session_id))
            reply = _reply_for_start_order_result(order)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        else:
            reply, messages = run_turn(
                messages,
                user_text=req.text,
                session_id=req.session_id,
                image_b64=req.image_b64,
                image_media_type=req.image_media_type,
            )

        store.save_session_messages(req.session_id, messages)
    except Exception:
        # Last-resort safety net: whatever went wrong (model, tool, network,
        # or saving session state), the user gets a friendly reply as a real
        # 200 instead of a raw 500 — a 500 skips this reply entirely and
        # surfaces the frontend's own generic network-error message instead,
        # which is a worse, less informative dead end. save_session_messages
        # is included in this try block for the same reason: it used to run
        # after this handler, unprotected.
        logger.exception("Unhandled error in /api/chat for session %s", req.session_id)
        return ChatResponse(reply=GENERIC_FAILURE_REPLY)

    return ChatResponse(reply=reply)


# Anchored on this file, like every other path in the repo (store.DATA_DIR,
# data_ingest.KB_DIR, ...). A bare "static" is resolved against the current
# working directory, so `uvicorn app.main:app` — or even just importing
# app.main — failed with "Directory 'static' does not exist" unless it was
# launched from the repo root.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
