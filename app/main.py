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


def _complete_pending_order(session_id: str, product_id: str, address_text: str) -> str:
    """Deterministically finishes an order once the user's next message supplies
    the address, instead of trusting the LLM to chain save_address + place_order
    + send_confirmation_email itself across turns (see llm_service/tools for why)."""
    email_in_text, address_only = _extract_email(address_text)
    store.save_address("demo_user", address_only or address_text)
    if email_in_text:
        store.save_email("demo_user", email_in_text)

    order = json.loads(tools.place_order(product_id))
    if "error" in order:
        return f"Sorry, something went wrong placing that order: {order['error']}"

    user = store.get_user("demo_user")
    recipient_email = user.get("email")
    summary = f"Order {order['order_id']}: {order['product_name']} (${order['price_usd']}) to {order['address']}"

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


def _resolve_bare_selection(text: str, last_products: list[dict] | None) -> dict | None:
    """After a product list is shown, users very naturally reply with just a
    number ("3") or an id fragment ("001") rather than the full product_id —
    and the model handles that unreliably (either declining it as unrelated,
    or hallucinating a completion for the wrong product). Resolves it
    deterministically against whatever list was actually just shown, matching
    by the product id's numeric suffix first (unambiguous even though ids
    like fev-001/col-001 collide numerically across categories, since we only
    ever look inside the specific list just shown), falling back to plain
    list position."""
    if not last_products:
        return None
    t = text.strip()
    if not re.fullmatch(r"\d{1,3}", t):
        return None
    n = int(t)
    for product in last_products:
        suffix = product["id"].split("-")[-1]
        if suffix.isdigit() and int(suffix) == n:
            return product
    if 1 <= n <= len(last_products):
        return last_products[n - 1]
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
}


def _is_affirmative(text: str) -> bool:
    """A bare "yes"/"ok"/"yes please" is the natural way to respond to
    "would you like to order this?"."""
    normalized = text.strip().lower().strip("!.,? ")
    words = re.findall(r"[a-z']+", normalized)
    return bool(words) and all(w in AFFIRMATIVE_WORDS for w in words)


def _reply_for_start_order_result(order: dict) -> str:
    if "error" in order:
        return f"Sorry, {order['error']}"

    if not order.get("order_placed"):
        # tools.start_order already recorded the pending order internally;
        # we just need to ask for the address ourselves.
        return "Sure! What's your shipping address so I can send that out?"

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
        elif pending_email_order_id and text and _looks_like_an_email(text):
            store.clear_pending_email(req.session_id)
            email_in_text, _ = _extract_email(text)
            reply = _complete_pending_email(email_in_text, pending_email_order_id)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_product_id and text and _looks_like_an_address(text):
            store.clear_pending_order(req.session_id)
            reply = _complete_pending_order(req.session_id, pending_product_id, text)
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_email_order_id and text:
            # Reply doesn't look like an email — handling this via run_turn
            # let the model free-generate instead of admitting it still
            # needs one; keep the LLM out of this state entirely rather
            # than risk it again.
            reply = "I still need an email address to send your confirmation to — what's the best one to use?"
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
        elif pending_product_id and text:
            # Reply doesn't look like an address — observed failure here:
            # the model, given a bare "yes" while an address was still
            # pending, fabricated a plausible-looking address out of thin
            # air, called the real save_address tool with it, and claimed
            # the order was placed — all without ever actually calling
            # start_order again. Keeping the LLM out of this state entirely
            # (like the pending-email branch above) removes the chance for
            # that to happen instead of hoping the model admits it needs
            # a real address.
            product = store.find_product(pending_product_id)
            product_name = product["name"] if product else "your order"
            reply = f"I still need your shipping address to complete the order for {product_name} — could you share it?"
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": reply})
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
    except Exception:
        # Last-resort safety net: whatever went wrong (model, tool, network),
        # the user gets a friendly reply instead of a raw 500, and the
        # conversation history isn't left in a half-updated state.
        logger.exception("Unhandled error in /api/chat for session %s", req.session_id)
        return ChatResponse(reply=GENERIC_FAILURE_REPLY)

    store.save_session_messages(req.session_id, messages)

    return ChatResponse(reply=reply)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
