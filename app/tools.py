import difflib
import json
import re
import smtplib
from email.message import EmailMessage
from email.utils import formataddr

from app import config, retrieval, store

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

# Individual words for typo-tolerant fallback (see _fuzzy_classify) — a
# misspelling like "runnig nose" won't contain the exact phrase "runny nose",
# so the plain substring check above misses it entirely.
FEVER_WORDS = {"fever", "temperature", "chills", "chill", "headache", "paracetamol", "acetaminophen", "ibuprofen"}
COLD_WORDS = {
    "cold", "runny", "nose", "sneezing", "sneeze", "congestion", "sore",
    "throat", "cough", "stuffy", "nasal", "flu", "cetirizine", "antihistamine",
    "pseudoephedrine", "decongestant", "dextromethorphan", "phenylephrine",
    "chlorpheniramine",
}

# Fuzzy (edit-distance) matching is ONLY safe for longer, distinctive words.
# Short words like "cold", "sore", "flu" have too many unrelated short-word
# neighbors (e.g. "code" vs "cold" scores *higher* than the real typo "runnig"
# vs "runny" does) — fuzzy-matching them caused "fix my code" to be
# misclassified as a cold symptom. Short words rely on exact membership only
# (still typo-tolerant for the common case: whole words split out of phrases,
# like "nose" catching "runnig nose" even though "runnig" itself doesn't match
# anything).
_LONG_FEVER_WORDS = {w for w in FEVER_WORDS if len(w) >= 7}
_LONG_COLD_WORDS = {w for w in COLD_WORDS if len(w) >= 7}
FUZZY_CUTOFF = 0.8
FUZZY_MIN_WORD_LEN = 7


def _category_hit(s: str, words: list[str], keywords: list[str], word_set: set[str], long_word_set: set[str]) -> bool:
    """Substring keyword match, or exact word membership, or (for long,
    distinctive words only — see FUZZY_MIN_WORD_LEN) a fuzzy match. All
    three checks always run for a category — this used to be structured as
    "try substrings for both categories, and only fall back to word/fuzzy
    matching if NEITHER substring matched at all", which meant a message
    like "nose block and feverih" (fever hits the "fever" substring inside
    "feverih", so fuzzy fallback never runs at all) never got to check "nose"
    as a cold word, silently dropping cold from the result entirely."""
    if any(k in s for k in keywords):
        return True
    if any(w in word_set for w in words):
        return True
    return any(
        len(w) >= FUZZY_MIN_WORD_LEN and difflib.get_close_matches(w, long_word_set, n=1, cutoff=FUZZY_CUTOFF)
        for w in words
    )


def classify_categories(symptom: str) -> list[str]:
    """Returns every category the text matches, not just the first — a
    message like "nose block and feverih" describes both a cold symptom and
    a fever, and both need to come back so the model can offer both, not
    just whichever one this function happened to check/match first."""
    s = symptom.strip().lower()
    if s == "fever":
        return ["fever"]
    if s == "cold":
        return ["cold"]

    words = re.findall(r"[a-z]+", s)
    categories = []
    if _category_hit(s, words, FEVER_KEYWORDS, FEVER_WORDS, _LONG_FEVER_WORDS):
        categories.append("fever")
    if _category_hit(s, words, COLD_KEYWORDS, COLD_WORDS, _LONG_COLD_WORDS):
        categories.append("cold")
    return categories


def classify(symptom: str) -> str | None:
    categories = classify_categories(symptom)
    return categories[0] if categories else None


def lookup_symptom(symptom: str) -> str:
    categories = classify_categories(symptom)
    if not categories:
        return json.dumps({
            "matched": False,
            "message": "No fever/cold match for this symptom — out of scope for this demo.",
        })
    products = [p for category in categories for p in store.lookup_by_category(category)]
    return json.dumps({
        "matched": True,
        "category": categories[0] if len(categories) == 1 else "+".join(categories),
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


# Temporary, hand-picked placeholder — NOT sourced from any real pharmacy/
# regulatory quantity guideline. Fetching genuine per-medicine max-quantity
# rules (they vary by drug, jurisdiction, and pack size) was out of scope for
# this demo's timeline; this exists so an order at least has *some* sane cap
# instead of none, and should be replaced with real guidance before this is
# ever more than a demo.
MAX_QUANTITY_PER_ORDER = 2


def _clamp_quantity(quantity: int) -> tuple[int, bool]:
    """Returns (clamped_quantity, was_clamped)."""
    quantity = max(1, quantity)
    if quantity > MAX_QUANTITY_PER_ORDER:
        return MAX_QUANTITY_PER_ORDER, True
    return quantity, False


def place_order(product_id: str, quantity: int = 1) -> str:
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
    quantity, clamped = _clamp_quantity(quantity)
    order = store.create_order(user_id="demo_user", product_id=product_id, address=address, quantity=quantity)
    if clamped:
        order["quantity_clamped"] = (
            f"Quantity was capped at our per-order limit of {MAX_QUANTITY_PER_ORDER} "
            f"for this product — mention this to the user."
        )
    return json.dumps(order)


def start_order(product_id: str, session_id: str, quantity: int = 1) -> str:
    """Starts placing an order — never completes it silently. If there's no
    address on file, defers to ask for one; if an address IS on file, defers
    to ask the user to confirm it's still correct rather than assuming it
    hasn't changed. Either way, the actual order is only created once the
    user's next message resolves the deferred step (see main.py's
    _complete_pending_order / _complete_confirmed_order). This collapsing of
    state-changing steps into deferred, deterministic follow-ups (rather than
    chaining separate save_address/place_order/send_confirmation_email tool
    calls) exists because that chaining proved unreliable with a small local
    model — it lost track of state across turns."""
    product = store.find_product(product_id)
    if product is None:
        return json.dumps({
            "error": f"Unknown product_id '{product_id}' — not in the fever/cold catalog."
        })

    quantity, clamped = _clamp_quantity(quantity)
    clamp_note = (
        f" (Note: capped at our per-order limit of {MAX_QUANTITY_PER_ORDER} — mention this to the user.)"
        if clamped else ""
    )

    address = store.get_address("demo_user")
    if not address:
        store.set_pending_order(session_id, product_id, quantity)
        return json.dumps({
            "order_placed": False,
            "message": (
                "No address on file. Ask the user for their shipping address in plain "
                "conversational text now, then stop — do not call any more tools this "
                f"turn. Their very next message will automatically be captured as the "
                f"address and used to complete this order.{clamp_note}"
            ),
        })

    store.set_pending_address_confirmation(session_id, product_id, quantity)
    return json.dumps({
        "order_placed": False,
        "needs_address_confirmation": True,
        "address_on_file": address,
        "message": (
            f'Ask the user to confirm this shipping address on file: "{address}". '
            "Do not call any more tools this turn. If they confirm (e.g. \"yes\"), "
            "their next message completes the order to that address automatically. "
            "If they give a different address instead, that new one is used instead "
            f"— never assume the address on file is still correct without asking.{clamp_note}"
        ),
    })


def _build_confirmation_note(order: dict) -> dict:
    recipient_email = store.get_email("demo_user")
    if recipient_email:
        summary = (
            f"Order {order['order_id']}: {order['quantity']}x {order['product_name']} "
            f"(${order['total_price_usd']}) to {order['address']}"
        )
        email_result = json.loads(send_confirmation_email(recipient_email, summary))
        order["email_sent"] = email_result.get("sent", False)
    else:
        order["email_sent"] = False
        order["email_needed"] = True
    return order


def complete_confirmed_order(session_id: str, product_id: str, quantity: int) -> str:
    """Finishes an order once the user has confirmed the address on file is
    still correct — the deterministic counterpart to start_order's deferral,
    called from main.py rather than left to the model (see start_order's
    docstring)."""
    order = json.loads(place_order(product_id, quantity))
    if "error" in order:
        return json.dumps(order)
    order["order_placed"] = True
    if not store.get_email("demo_user"):
        store.set_pending_email(session_id, order["order_id"])
    return json.dumps(_build_confirmation_note(order))


def send_confirmation_email(to: str, order_details: str) -> str:
    if not config.SMTP_CONFIGURED:
        print(f"[MOCK EMAIL] would send to {to}: {order_details}")
        return json.dumps({"sent": True, "mode": "mock"})

    message = EmailMessage()
    message["Subject"] = "Your MedleyMed order confirmation"
    message["From"] = formataddr((config.SMTP_FROM_NAME, config.SMTP_USER))
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


def lookup_medicine_info(query: str) -> str:
    """RAG lookup: embeds the query, retrieves the most relevant chunks from
    the local knowledge base (see retrieval.py/data_ingest.py), and returns
    them for the model to ground its answer in — including the source file,
    so the model can cite it rather than answering from unverified general
    knowledge."""
    results = retrieval.search(query)
    if not results:
        return json.dumps({
            "results": [],
            "message": "No information found in our knowledge base for that.",
        })
    return json.dumps({"results": results})


TOOL_FUNCTIONS = {
    "lookup_symptom": lambda args: lookup_symptom(args.get("symptom", "")),
    "get_saved_address": lambda args: get_saved_address(args.get("user_id", "demo_user")),
    "save_address": lambda args: save_address(
        args.get("user_id", "demo_user"), args.get("address", "")
    ),
    "lookup_medicine_info": lambda args: lookup_medicine_info(args.get("query", "")),
}
