import json
import smtplib
from email.message import EmailMessage

from app import config, store

FEVER_KEYWORDS = [
    "fever", "temperature", "chills", "body ache", "headache",
    "paracetamol", "acetaminophen", "ibuprofen",
]
COLD_KEYWORDS = [
    "cold", "runny nose", "sneezing", "congestion", "sore throat",
    "cough", "stuffy", "nasal", "flu",
    "cetirizine", "antihistamine", "pseudoephedrine", "decongestant",
    "dextromethorphan", "phenylephrine", "chlorpheniramine",
]


def classify(symptom: str) -> str | None:
    s = symptom.strip().lower()
    if s in ("fever", "cold"):
        return s
    if any(k in s for k in FEVER_KEYWORDS):
        return "fever"
    if any(k in s for k in COLD_KEYWORDS):
        return "cold"
    return None


def lookup_symptom(symptom: str) -> str:
    category = classify(symptom)
    if category is None:
        return json.dumps({
            "matched": False,
            "message": "No fever/cold match for this symptom — out of scope for this demo.",
        })
    products = store.lookup_by_category(category)
    return json.dumps({
        "matched": True,
        "category": category,
        "products": [
            {
                "id": p["id"],
                "name": p["name"],
                "active_ingredient": p["active_ingredient"],
                "price_usd": p["price_usd"],
                "description": p["description"],
            }
            for p in products
        ],
    })


def get_saved_address(user_id: str) -> str:
    address = store.get_address(user_id)
    return json.dumps({"address": address})


def save_address(user_id: str, address: str) -> str:
    store.save_address(user_id, address)
    return json.dumps({"saved": True, "address": address})


def place_order(product_id: str) -> str:
    product = store.find_product(product_id)
    if product is None:
        return json.dumps({
            "error": f"Unknown product_id '{product_id}' — not in the fever/cold catalog."
        })
    address = store.get_address("demo_user")
    if not address:
        return json.dumps({
            "error": "No address on file. Ask the user for their shipping address, "
                     "call save_address with it, then call place_order again."
        })
    order = store.create_order(user_id="demo_user", product_id=product_id, address=address)
    return json.dumps(order)


def start_order(product_id: str, session_id: str) -> str:
    """Atomically completes an order if an address is on file, or defers it by
    marking the session as awaiting an address. This exists because chaining
    separate save_address/place_order/send_confirmation_email tool calls proved
    unreliable with a small local model — collapsing the state-changing steps
    into one call (or one deferred, deterministic follow-up) removes the chance
    for the model to lose track of state across turns."""
    product = store.find_product(product_id)
    if product is None:
        return json.dumps({
            "error": f"Unknown product_id '{product_id}' — not in the fever/cold catalog."
        })

    address = store.get_address("demo_user")
    if not address:
        store.set_pending_order(session_id, product_id)
        return json.dumps({
            "order_placed": False,
            "message": (
                "No address on file. Ask the user for their shipping address in plain "
                "conversational text now, then stop — do not call any more tools this "
                "turn. Their very next message will automatically be captured as the "
                "address and used to complete this order."
            ),
        })

    order = json.loads(place_order(product_id))
    if "error" in order:
        return json.dumps(order)

    user = store.get_user("demo_user")
    summary = f"Order {order['order_id']}: {order['product_name']} (${order['price_usd']}) to {order['address']}"
    email_result = json.loads(send_confirmation_email(user["email"], summary))
    order["order_placed"] = True
    order["email_sent"] = email_result.get("sent", False)
    return json.dumps(order)


def send_confirmation_email(to: str, order_details: str) -> str:
    if not config.SMTP_CONFIGURED:
        print(f"[MOCK EMAIL] would send to {to}: {order_details}")
        return json.dumps({"sent": True, "mode": "mock"})

    message = EmailMessage()
    message["Subject"] = "Your order confirmation"
    message["From"] = config.SMTP_USER
    message["To"] = to
    message.set_content(order_details)

    try:
        with smtplib.SMTP(config.SMTP_HOST, int(config.SMTP_PORT), timeout=10) as smtp:
            smtp.starttls()
            smtp.login(config.SMTP_USER, config.SMTP_PASS)
            smtp.send_message(message)
        return json.dumps({"sent": True, "mode": "smtp"})
    except Exception as exc:
        # An order should never fail just because email delivery did — log it
        # and let the caller decide how to tell the user, rather than raising.
        print(f"[EMAIL ERROR] could not send to {to}: {exc}")
        return json.dumps({"sent": False, "mode": "error", "error": str(exc)})


TOOL_FUNCTIONS = {
    "lookup_symptom": lambda args: lookup_symptom(args.get("symptom", "")),
    "get_saved_address": lambda args: get_saved_address(args.get("user_id", "demo_user")),
    "save_address": lambda args: save_address(
        args.get("user_id", "demo_user"), args.get("address", "")
    ),
}
