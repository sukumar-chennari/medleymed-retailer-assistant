"""A seeded stateful fuzz of the deterministic chat flows.

Random 9-message sessions are drawn from a pool of realistic messages (symptoms,
orders, quantities, addresses, emails, cancels, questions, filler, junk) with the
chat model unavailable, and after EVERY turn these invariants are checked:

- an order is created exactly when the reply says "Order confirmed!";
- a cancelled order never comes back;
- the saved address is never an instruction, filler, an email, a product name or
  an order id;
- pending orders and confirmations name a real product with a quantity of 1-2;
- every order row is valid; every reply is non-empty and has no traceback;
- consistency: every receipt (id, product, quantity, total, address) matches its
  database row, "has been cancelled" is true, an order listing shows every order
  with its real status, an address/email read-back equals the stored value, and
  remembered lists / email prompts / recommendations reference real things;
- safety: no ibuprofen order is ever created after a stated ulcer, ibuprofen
  allergy, warfarin or pregnancy; no pseudoephedrine order after stated high blood
  pressure; an overdose, an infant's fever and "I can't breathe" always get the
  escalation; a saved email is always a valid email.

It found a real bug on its first run (Nov 2): "hmm", "later" and "..." were saved
as the shipping address and the order placed to them. The seed is fixed so a
failure is reproducible; raise SEQUENCES locally to search harder."""

import os
import random
import re

import pytest
from fastapi.testclient import TestClient

from app import agent, config, main, store

# Overridable for a bigger local search: FUZZ_SEQUENCES=1000 FUZZ_SEED=3 pytest tests/test_state_invariants.py
SEQUENCES = int(os.environ.get("FUZZ_SEQUENCES", 70))
TURNS = 9
SEED = int(os.environ.get("FUZZ_SEED", 7))
client = TestClient(main.app)

POOL = [
    "I have a fever", "I have a fever and a dry cough", "I'm an adult with a sore throat", "my 4 year old has a cold",
    "adult", "child", "for myself", "dry", "wet", "yes", "no", "ok", "1", "2", "3", "the first one", "the last one",
    "both of them", "order Ibuprofen 200mg Tablets", "order 2 Cetirizine 10mg Antihistamine Tablets", "order Tylenol",
    "order col-004", "make it 2", "actually just 1", "10 High St, Leeds", "5 Mid Rd, Pune", "Flat 4, Rose Lane, Delhi",
    "a@b.com", "my email is x@y.com", "cancel my last order", "cancel all my orders", "cancel ord-0001", "reorder that",
    "one more of the same", "show my orders", "what is my address", "change my address to 99 New Rd, York",
    "show me all your products", "how much is ibuprofen", "compare paracetamol and ibuprofen",
    "can I take paracetamol and ibuprofen together", "do you have ibuprofen", "help", "start over", "later", "skip it",
    "send me the receipt", "I have a stomach ulcer", "I'm allergic to ibuprofen", "I took 10 paracetamol tablets",
    "order Ibuprofen 200mg Tablets ship to 5 Mid Rd, Pune", "how long does delivery take", "who are you", "hmm", "what",
    "...", "thanks", "order ibuprofen 400mg", "?!", "💊", "not now", "ordr ibuprofin 200 mg", "मुझे बुखार है",
    "I have high blood pressure", "I'm pregnant", "I'm on warfarin", "I have asthma", "order Cold & Flu Multi-Symptom Relief",
    "order Pseudoephedrine Decongestant Tablets", "my baby is 2 months old and has a fever", "I can't breathe",
    "what is the difference between the first two", "do you have Tylenol", "cancel", "order paracetamol 500mg x2",
    "I am 25 years old", "my wife has a fever and a cough", "naaku jvaram undi", "is cetirizine available", "hello",
    "good morning", "bye", "I want to order", "order it", "same again", "I don't want it",
]
# Short risky patterns spliced into the random sessions. Pure random messages
# rarely line up "start an order -> say something -> complete it", and that is
# exactly where the bugs found so far lived (a condition or a filler word said
# while an order waited), so the generator inserts these on purpose.
RISKY_CHUNKS = [
    ["order Ibuprofen 200mg Tablets", "I'm on warfarin", "5 Mid Rd, Pune"],
    ["order Ibuprofen 200mg Tablets", "I have a stomach ulcer", "yes"],
    ["order Ibuprofen 200mg Tablets", "I'm pregnant", "10 High St, Leeds"],
    ["I'm allergic to ibuprofen", "order Ibuprofen 200mg Tablets", "5 Mid Rd, Pune"],
    ["order Pseudoephedrine Decongestant Tablets", "I have high blood pressure", "Flat 4, Rose Lane, Delhi"],
    ["order Ibuprofen 200mg Tablets", "hmm", "later"],
    ["order Ibuprofen 200mg Tablets", "...", "5 Mid Rd, Pune"],
    ["order col-004", "I can't breathe", "10 High St, Leeds"],
    ["order Ibuprofen 200mg Tablets", "I took 10 paracetamol tablets", "5 Mid Rd, Pune"],
    ["order Paracetamol 500mg Tablets", "a@b.com", "5 Mid Rd, Pune"],
    ["order Ibuprofen 200mg Tablets", "make it 2", "5 Mid Rd, Pune"],
    ["order Ibuprofen 200mg Tablets", "I have a stomach ulcer", "5 Mid Rd, Pune"],
]
# The only real addresses the pool ever types. Anything else saved as the shipping
# address is junk: a stronger check than a list of banned words, which let "dry" and
# "order it" through (found Nov 5 when a deliberately broken receipt exposed them).
REAL_ADDRESSES = {"10 High St, Leeds", "5 Mid Rd, Pune", "Flat 4, Rose Lane, Delhi", "99 New Rd, York"}
_FILLER_ADDRESSES = {"yes", "no", "ok", "help", "later", "hmm", "what", "...", "thanks", "?!", "stop"}


@pytest.fixture(autouse=True)
def _no_model(monkeypatch, isolated_db):
    monkeypatch.setattr(config, "SMTP_CONFIGURED", False)

    def _unavailable(*args, **kwargs):
        raise RuntimeError("the chat model is unavailable in this test")

    monkeypatch.setattr(agent, "create_agent", _unavailable)


def _violations(session_id, message, reply, orders_before, history_text):
    found = []
    catalog = store.get_catalog()
    ids = {p["id"] for p in catalog}
    orders = store.list_orders("demo_user")
    before_ids = {o["order_id"] for o in orders_before}
    new = [o for o in orders if o["order_id"] not in before_ids]
    confirmed = "Order confirmed!" in reply
    if confirmed and len(new) != 1:
        found.append(f"'Order confirmed!' with {len(new)} new orders")
    if new and not confirmed:
        found.append(f"order created silently: {[o['order_id'] for o in new]}")
    for order in orders:
        previous = next((b for b in orders_before if b["order_id"] == order["order_id"]), None)
        if previous and previous["status"] == "cancelled" and order["status"] != "cancelled":
            found.append(f"cancelled order revived: {order['order_id']}")
        if not 1 <= order["quantity"] <= 2 or order["product_id"] not in ids:
            found.append(f"invalid order row: {order}")
    address = store.get_address("demo_user")
    if address and address not in REAL_ADDRESSES:
        found.append(f"saved address is not one the customer ever gave: {address!r}")
    if address:
        low = address.lower()
        if (
            re.search(r"\bord-\d+\b|\b(?:fev|col)-\d{3}\b", low) or "@" in address or low in _FILLER_ADDRESSES
            or any(p["name"].lower() in low for p in catalog)
        ):
            found.append(f"bad saved address: {address!r}")
    for label, pending in (("pending order", store.get_pending_order(session_id)),
                           ("pending confirmation", store.get_pending_address_confirmation(session_id))):
        if pending and (pending["product_id"] not in ids or not 1 <= pending["quantity"] <= 2):
            found.append(f"invalid {label}: {pending}")
    lowered = message.lower()
    # Only what the application actually KEPT counts as "said": with the chat model
    # unavailable here, a statement that goes to the model ("I'm pregnant") is not
    # saved, so no code could possibly act on it.
    kept = [m["content"] for m in store.get_session_messages(session_id) if m.get("role") == "user" and isinstance(m.get("content"), str)]
    said = " ".join([*kept, message]).lower()
    new_products = [o["product_id"] for o in new]
    stated_ibuprofen_stopper = re.search(
        r"\b(?:stomach|peptic|gastric)\s+ulcer|\ballergic to ibuprofen|\bon warfarin|\bi'?m pregnant", said
    )
    if "fev-003" in new_products and stated_ibuprofen_stopper:
        found.append(f"ibuprofen ordered despite a stated condition: {stated_ibuprofen_stopper.group(0)!r}")
    if "col-002" in new_products and "high blood pressure" in said:
        found.append("pseudoephedrine ordered despite stated high blood pressure")
    if "i took 10 paracetamol" in lowered and "emergency number" not in reply:
        found.append(f"overdose message without the emergency reply: {reply[:60]!r}")
    if "can't breathe" in lowered and "emergency" not in reply.lower():
        found.append(f"emergency symptom without escalation: {reply[:60]!r}")
    if "2 months old and has a fever" in lowered and "under 3 months" not in reply:
        found.append(f"infant fever without escalation: {reply[:60]!r}")
    email = store.get_email("demo_user")
    if email and not re.fullmatch(r"[\w.+-]+@[\w-]+(?:\.[\w-]*\w)+", email):
        found.append(f"bad saved email: {email!r}")
    found.extend(_consistency_problems(session_id, message, reply, orders))
    if not reply.strip() or "Traceback" in reply:
        found.append(f"bad reply: {reply[:60]!r}")
    return found


def _consistency_problems(session_id, message, reply, orders):
    """Every number and name the assistant states must match the database."""
    found = []
    by_id = {o["order_id"]: o for o in orders}
    catalog = {p["id"]: p for p in store.get_catalog()}
    # a receipt block: Order ID / Product / Quantity / Total / Shipping to
    for block in re.finditer(
        r"Order ID: (ord-\d+)\nProduct: ([^\n]+)\nQuantity: (\d+)\nTotal: \$([\d.]+)\nShipping to: ([^\n]+)", reply
    ):
        order_id, product, quantity, total, address = block.groups()
        row = by_id.get(order_id)
        if row is None:
            found.append(f"receipt for an order that does not exist: {order_id}")
            continue
        price = catalog[row["product_id"]]["price_usd"]
        if product != catalog[row["product_id"]]["name"]:
            found.append(f"receipt product {product!r} != row {catalog[row['product_id']]['name']!r}")
        if int(quantity) != row["quantity"]:
            found.append(f"receipt quantity {quantity} != row {row['quantity']}")
        if abs(float(total) - round(price * row["quantity"], 2)) > 0.011:
            found.append(f"receipt total {total} != {price} x {row['quantity']}")
        if address != row["address"]:
            found.append(f"receipt address {address!r} != row {row['address']!r}")
    # "Order ord-0002 (Name) has been cancelled." must really be cancelled
    for cancelled in re.finditer(r"Order (ord-\d+) \([^)]*\) has been cancelled\.", reply):
        row = by_id.get(cancelled.group(1))
        if row is None or row["status"] != "cancelled":
            found.append(f"claims {cancelled.group(1)} was cancelled but the row says {row and row['status']}")
    # an order listing must show every order with its real status
    if reply.startswith("Here is your order") or reply.startswith("Here are your orders"):
        for order in orders:
            if f"Order ID: {order['order_id']}" not in reply:
                found.append(f"order listing is missing {order['order_id']}")
            elif f"Status: {order['status']}" not in reply.split(f"Order ID: {order['order_id']}")[1].split("Order ID:")[0]:
                found.append(f"order listing shows the wrong status for {order['order_id']}")
    # reading the address back must give the stored address
    if reply.startswith("The shipping address I have on file for you is:"):
        if reply.split(": ", 1)[1].strip() != store.get_address("demo_user"):
            found.append("address read-back differs from the stored address")
    if reply.startswith("The email address I have on file for you is:"):
        if reply.split(": ", 1)[1].strip() != store.get_email("demo_user"):
            found.append("email read-back differs from the stored email")
    # every "(fev-001)" the assistant writes must come right after that product's real name
    for marker in re.finditer(r"\(((?:fev|col)-\d{3})\)", reply):
        product_id = marker.group(1)
        before = reply[max(0, marker.start() - 90): marker.start()].rstrip()
        real = catalog.get(product_id)
        if real is None or not before.endswith(real["name"]):
            found.append(f"id ({product_id}) not preceded by its product name {real and real['name']!r}: ...{before[-50:]!r}")
    # a line that names exactly one product and states a price must state ITS price
    names = {p["name"]: p for p in catalog.values()}
    for line in reply.split("\n"):
        if "$" not in line or line.startswith(("Total:", "Quantity:")):
            continue
        in_line = [p for n, p in names.items() if n in line]
        # ignore a name that is only part of a longer product name in the same line
        in_line = [p for p in in_line if not any(p["name"] != q["name"] and p["name"] in q["name"] for q in in_line)]
        if len(in_line) == 1:
            product = in_line[0]
            for amount in re.findall(r"\$(\d+\.\d{2})", line):
                if abs(float(amount) - product["price_usd"]) > 0.011 and "Total" not in line:
                    found.append(f"price {amount} stated for {product['name']} (catalog {product['price_usd']}): {line.strip()[:100]!r}")
    # remembered lists and prompts must reference real things
    for product in store.get_last_products(session_id) or []:
        if product["id"] not in catalog:
            found.append(f"remembered a product that is not in the catalog: {product['id']}")
    pending_email = store.get_pending_email(session_id)
    if pending_email and pending_email not in by_id:
        found.append(f"pending email prompt for an order that does not exist: {pending_email}")
    last = store.get_last_recommended_product(session_id)
    if last and last not in catalog:
        found.append(f"last recommended product is not in the catalog: {last}")
    return found


def test_the_invariants_hold_across_random_sessions():
    rng = random.Random(SEED)
    failures = []
    for sequence in range(SEQUENCES):
        store.reset_demo_data()
        session_id = f"fuzz-{sequence}"
        history = []
        plan = [rng.choice(POOL) for _ in range(TURNS)]
        if rng.random() < 0.6:
            at = rng.randrange(0, TURNS - 2)
            plan[at: at + 3] = rng.choice(RISKY_CHUNKS)
        for message in plan:
            history.append(message)
            before = store.list_orders("demo_user")
            res = client.post("/api/chat", json={"session_id": session_id, "text": message})
            reply = res.json().get("reply", "") if res.status_code == 200 else f"HTTP {res.status_code}"
            for problem in _violations(session_id, message, reply, before, history):
                failures.append(f"{problem}  | after {history!r}")
    assert failures == [], "\n".join(failures[:5])


def _session_snapshot(session_id):
    """Everything the application remembers about ONE conversation."""
    return {
        "pending_order": store.get_pending_order(session_id),
        "pending_address_confirmation": store.get_pending_address_confirmation(session_id),
        "pending_clarification": store.get_pending_clarification(session_id),
        "pending_email": store.get_pending_email(session_id),
        "last_products": [p["id"] for p in (store.get_last_products(session_id) or [])],
        "last_recommended": store.get_last_recommended_product(session_id),
    }


def test_two_interleaved_sessions_do_not_interfere():
    """Two customers chat at once. The address, email and orders are shared by
    design (a single demo user), but what a conversation is WAITING for is its own:
    a turn in one session must never change another session's pending order,
    address confirmation, open question, email prompt or remembered list. All the
    single-session invariants must hold for the session that spoke as well."""
    rng = random.Random(SEED + 1)
    failures = []
    sequences = max(20, SEQUENCES // 2)
    for sequence in range(sequences):
        store.reset_demo_data()
        sessions = [f"two-{sequence}-a", f"two-{sequence}-b"]
        histories = {s: [] for s in sessions}
        plan = [rng.choice(POOL) for _ in range(TURNS + 3)]
        if rng.random() < 0.6:
            at = rng.randrange(0, TURNS)
            plan[at: at + 3] = rng.choice(RISKY_CHUNKS)
        for message in plan:
            speaker = rng.choice(sessions)
            other = sessions[1] if speaker == sessions[0] else sessions[0]
            histories[speaker].append(message)
            before_orders = store.list_orders("demo_user")
            other_before = _session_snapshot(other)
            res = client.post("/api/chat", json={"session_id": speaker, "text": message})
            reply = res.json().get("reply", "") if res.status_code == 200 else f"HTTP {res.status_code}"
            for problem in _violations(speaker, message, reply, before_orders, histories[speaker]):
                failures.append(f"{problem}  | {speaker[-1]} said {histories[speaker]!r}")
            other_after = _session_snapshot(other)
            if other_after != other_before:
                changed = {k: (other_before[k], other_after[k]) for k in other_before if other_before[k] != other_after[k]}
                failures.append(f"session {other[-1]} changed by session {speaker[-1]}'s {message!r}: {changed}")
    assert failures == [], "\n".join(failures[:5])


# Messages that only ASK for information. They may change what is remembered for
# conversation flow (the list on screen), but they must never create or change an
# order, change the saved address or email, or disturb an order that is waiting.
NEUTRAL_MESSAGES = [
    "how much is ibuprofen", "what is the difference between paracetamol and ibuprofen", "compare paracetamol and ibuprofen",
    "can I take paracetamol and ibuprofen together", "show me all your products", "who are you", "what can you do", "are you a doctor",
    "what is my address", "show my orders", "how long does delivery take", "what is your refund policy", "do you have Tylenol",
    "is cetirizine available", "what is the cheapest cold medicine", "can you speak telugu", "what is dextromethorphan",
    "how many tablets in a pack of paracetamol", "is cetirizine safe during pregnancy", "which is better, ibuprofen or paracetamol",
]


def _durable_state(session_id):
    return {
        "orders": sorted((o["order_id"], o["status"], o["quantity"], o["address"]) for o in store.list_orders("demo_user")),
        "address": store.get_address("demo_user"),
        "email": store.get_email("demo_user"),
        "pending_order": store.get_pending_order(session_id),
        "pending_address_confirmation": store.get_pending_address_confirmation(session_id),
    }


def test_questions_never_change_orders_addresses_or_what_is_waiting():
    rng = random.Random(SEED + 2)
    failures = []
    for sequence in range(max(30, SEQUENCES // 2)):
        store.reset_demo_data()
        session_id = f"neutral-{sequence}"
        plan = [rng.choice(POOL) for _ in range(TURNS)]
        if rng.random() < 0.6:
            at = rng.randrange(0, TURNS - 2)
            plan[at: at + 3] = rng.choice(RISKY_CHUNKS)
        # a neutral question after every other message, on purpose
        steps = []
        for message in plan:
            steps.append(message)
            steps.append(rng.choice(NEUTRAL_MESSAGES))
        history = []
        for message in steps:
            history.append(message)
            before = _durable_state(session_id)
            res = client.post("/api/chat", json={"session_id": session_id, "text": message})
            reply = res.json().get("reply", "") if res.status_code == 200 else f"HTTP {res.status_code}"
            after = _durable_state(session_id)
            if message in NEUTRAL_MESSAGES and before != after:
                changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
                failures.append(f"a question changed state: {message!r} -> {changed}  | after {history[-6:]!r}")
            if not reply.strip():
                failures.append(f"empty reply to {message!r}")
    assert failures == [], "\n".join(failures[:5])
