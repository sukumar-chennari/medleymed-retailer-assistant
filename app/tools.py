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


def _fuzzy_classify(s: str) -> str | None:
    words = re.findall(r"[a-z]+", s)

    fever_hit = any(w in FEVER_WORDS for w in words)
    cold_hit = any(w in COLD_WORDS for w in words)

    if not fever_hit:
        fever_hit = any(
            len(w) >= FUZZY_MIN_WORD_LEN and difflib.get_close_matches(w, _LONG_FEVER_WORDS, n=1, cutoff=FUZZY_CUTOFF)
            for w in words
        )
    if not cold_hit:
        cold_hit = any(
            len(w) >= FUZZY_MIN_WORD_LEN and difflib.get_close_matches(w, _LONG_COLD_WORDS, n=1, cutoff=FUZZY_CUTOFF)
            for w in words
        )

    if fever_hit and not cold_hit:
        return "fever"
    if cold_hit and not fever_hit:
        return "cold"
    if fever_hit and cold_hit:
        return "fever"
    return None


def classify(symptom: str) -> str | None:
    s = symptom.strip().lower()
    if s in ("fever", "cold"):
        return s
    if any(k in s for k in FEVER_KEYWORDS):
        return "fever"
    if any(k in s for k in COLD_KEYWORDS):
        return "cold"
    return _fuzzy_classify(s)


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

    order["order_placed"] = True
    recipient_email = store.get_email("demo_user")

    if recipient_email:
        summary = f"Order {order['order_id']}: {order['product_name']} (${order['price_usd']}) to {order['address']}"
        email_result = json.loads(send_confirmation_email(recipient_email, summary))
        order["email_sent"] = email_result.get("sent", False)
    else:
        # No email on file — the order is still valid (address is all that's
        # required to ship), but there's no one to mail a confirmation to.
        # Mirrors the same missing-email handling in main.py's deterministic
        # address-completion path, for the fast "address already saved" case.
        store.set_pending_email(session_id, order["order_id"])
        order["email_sent"] = False
        order["email_needed"] = (
            "No email on file. Tell the user their order is confirmed, and ask "
            "for their email if they'd like a confirmation sent — their next "
            "message will be captured as the email automatically."
        )

    return json.dumps(order)


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
