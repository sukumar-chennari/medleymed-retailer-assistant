import json
import logging
import re

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app import store, tools
from app.llm_service import run_turn
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger("medicine_assistant")

app = FastAPI()

GENERIC_FAILURE_REPLY = (
    "Sorry, I ran into a problem processing that. Please try again — if it "
    "keeps happening, check that Ollama is running."
)


DISCLAIMER = (
    "This is general OTC guidance, not a medical diagnosis — consult a doctor if "
    "symptoms persist or worsen."
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
    "actually never mind" gets silently saved and used as the shipping address."""
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
        email_note = (
            f"A confirmation email has been sent to {recipient_email}."
            if email_result.get("sent")
            else "Your order is placed, though the confirmation email couldn't be sent."
        )
    else:
        # No email on file at all — the order still ships fine (address is
        # all that requires), but ask for an email so a confirmation can go
        # out; the next message is captured as the email, same pattern as
        # the address flow above.
        store.set_pending_email(session_id, order["order_id"])
        email_note = (
            "I don't have an email on file for you — reply with your email "
            "address if you'd like a confirmation sent."
        )

    return (
        f"Order confirmed!\n\n"
        f"Order ID: {order['order_id']}\n"
        f"Product: {order['product_name']}\n"
        f"Price: ${order['price_usd']}\n"
        f"Shipping to: {order['address']}\n\n"
        f"{email_note}\n\n{DISCLAIMER}"
    )


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


AFFIRMATIVE_PHRASES = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please",
    "go ahead", "do it", "confirm", "place it", "order it", "place the order",
}


def _is_affirmative(text: str) -> bool:
    """A bare "yes"/"ok" is the natural way to respond to "would you like to
    order this?" — an exact-match list (not fuzzy/word-based) keeps this from
    ever misfiring on a real sentence that happens to contain "ok"."""
    return text.strip().lower().strip("!.,? ") in AFFIRMATIVE_PHRASES


def _reply_for_start_order_result(order: dict) -> str:
    if "error" in order:
        return f"Sorry, {order['error']}"

    if not order.get("order_placed"):
        # tools.start_order already recorded the pending order internally;
        # we just need to ask for the address ourselves.
        return "Sure! What's your shipping address so I can send that out?"

    email_note = (
        "A confirmation email has been sent."
        if order.get("email_sent")
        else "I don't have an email on file for you — reply with your email address if you'd like a confirmation sent."
    )
    return (
        f"Order confirmed!\n\n"
        f"Order ID: {order['order_id']}\n"
        f"Product: {order['product_name']}\n"
        f"Price: ${order['price_usd']}\n"
        f"Shipping to: {order['address']}\n\n"
        f"{email_note}\n\n{DISCLAIMER}"
    )


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
        last_products = store.get_last_products(req.session_id)
        last_recommended_product_id = store.get_last_recommended_product(req.session_id)
        selected_product = _resolve_bare_selection(text, last_products) if text else None

        if pending_email_order_id and text and _looks_like_an_email(text):
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
