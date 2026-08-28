import json
import logging
import re

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app import guardrails, store, tools
from app.agent import resolve_clarification, run_turn
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger("medicine_assistant")

app = FastAPI()

GENERIC_FAILURE_REPLY = (
    "Sorry, I ran into a problem processing that. Please try again — if it "
    "keeps happening, check that Ollama is running."
)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


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


@app.get("/api/catalog")
def catalog():
    return store.get_catalog()


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
    remainder = remainder.strip(" ,;:-")
    return email, remainder


def _looks_like_an_email(text: str) -> bool:
    return EMAIL_RE.search(text) is not None


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


def _looks_like_an_address(text: str) -> bool:
    """A pending order treats the user's next message as their shipping address —
    but only if it plausibly is one. Without this, a reply like "thank you" or
    "actually never mind" gets silently saved and used as the shipping address.
    Also rejects anything containing a product id pattern (fev-001, col-002) —
    a message like "order col-001" has a digit and is long enough to pass the
    basic check, but is clearly a product reference, not an address."""
    if guardrails.PRODUCT_ID_RE.search(text):
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


def _wants_different_address(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
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

    summary = f"Order {order['order_id']}: {order['product_name']} (${order['price_usd']}) to {order['address']}"
    email_result = json.loads(tools.send_confirmation_email(email, summary))
    if email_result.get("sent"):
        return f"Thanks! I've sent the confirmation for order {order['order_id']} to {email}."
    return f"I've saved {email}, but the confirmation email couldn't be sent right now."


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.text and not req.image_b64:
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
        elif pending_address_confirmation and text and _wants_different_address(text):
            # They've signaled they want to give a new one but haven't
            # supplied it yet — ask specifically for it instead of repeating
            # the same compound question. pending_address_confirmation stays
            # set, so their very next message (the actual address) is still
            # handled by the _looks_like_an_address branch above.
            reply = "No problem — what's the new shipping address?"
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_address_confirmation and text:
            # Neither a clear yes, a new address, nor a rejection — ask again
            # rather than guessing which one they meant; keep the LLM out of
            # this state for the same reasons as the other pending branches.
            address = store.get_address("demo_user")
            reply = f"Just to confirm — should I ship to {address}, or would you like to give a different address?"
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
        elif pending_email_order_id and text and _declines(text):
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
        elif pending_product_id and text and _declines(text):
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")
