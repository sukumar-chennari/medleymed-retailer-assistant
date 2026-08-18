import json
import logging

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


def _complete_pending_order(session_id: str, product_id: str, address_text: str) -> str:
    """Deterministically finishes an order once the user's next message supplies
    the address, instead of trusting the LLM to chain save_address + place_order
    + send_confirmation_email itself across turns (see llm_service/tools for why)."""
    store.save_address("demo_user", address_text)
    order = json.loads(tools.place_order(product_id))

    if "error" in order:
        return f"Sorry, something went wrong placing that order: {order['error']}"

    user = store.get_user("demo_user")
    summary = f"Order {order['order_id']}: {order['product_name']} (${order['price_usd']}) to {order['address']}"
    email_result = json.loads(tools.send_confirmation_email(user["email"], summary))
    email_note = (
        "A confirmation email has been sent."
        if email_result.get("sent")
        else "Your order is placed, though the confirmation email couldn't be sent."
    )

    return (
        f"Order confirmed!\n\n"
        f"Order ID: {order['order_id']}\n"
        f"Product: {order['product_name']}\n"
        f"Price: ${order['price_usd']}\n"
        f"Shipping to: {order['address']}\n\n"
        f"{email_note}\n\n{DISCLAIMER}"
    )


def _looks_like_an_address(text: str) -> bool:
    """A pending order treats the user's next message as their shipping address —
    but only if it plausibly is one. Without this, a reply like "thank you" or
    "actually never mind" gets silently saved and used as the shipping address."""
    return any(ch.isdigit() for ch in text) and len(text) >= 8


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.text and not req.image_b64:
        raise HTTPException(status_code=400, detail="Provide text or an image.")

    messages = store.get_session_messages(req.session_id)

    try:
        pending_product_id = store.get_pending_order(req.session_id)

        if pending_product_id and req.text.strip() and _looks_like_an_address(req.text.strip()):
            store.clear_pending_order(req.session_id)
            reply = _complete_pending_order(req.session_id, pending_product_id, req.text.strip())
            messages.append({"role": "user", "content": req.text.strip()})
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
