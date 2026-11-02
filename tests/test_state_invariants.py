"""A seeded stateful fuzz of the deterministic chat flows.

Random 9-message sessions are drawn from a pool of realistic messages (symptoms,
orders, quantities, addresses, emails, cancels, questions, filler, junk) with the
chat model unavailable, and after EVERY turn these invariants are checked:

- an order is created exactly when the reply says "Order confirmed!";
- a cancelled order never comes back;
- the saved address is never an instruction, filler, an email, a product name or
  an order id;
- pending orders and confirmations name a real product with a quantity of 1-2;
- every order row is valid; every reply is non-empty and has no traceback.

It found a real bug on its first run (Nov 2): "hmm", "later" and "..." were saved
as the shipping address and the order placed to them. The seed is fixed so a
failure is reproducible; raise SEQUENCES locally to search harder."""

import random
import re

import pytest
from fastapi.testclient import TestClient

from app import agent, config, main, store

SEQUENCES = 120
TURNS = 9
SEED = 7
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
]
_FILLER_ADDRESSES = {"yes", "no", "ok", "help", "later", "hmm", "what", "...", "thanks", "?!", "stop"}


@pytest.fixture(autouse=True)
def _no_model(monkeypatch, isolated_db):
    monkeypatch.setattr(config, "SMTP_CONFIGURED", False)

    def _unavailable(*args, **kwargs):
        raise RuntimeError("the chat model is unavailable in this test")

    monkeypatch.setattr(agent, "create_agent", _unavailable)


def _violations(session_id, message, reply, orders_before):
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
    if not reply.strip() or "Traceback" in reply:
        found.append(f"bad reply: {reply[:60]!r}")
    return found


def test_the_invariants_hold_across_random_sessions():
    rng = random.Random(SEED)
    failures = []
    for sequence in range(SEQUENCES):
        store.reset_demo_data()
        session_id = f"fuzz-{sequence}"
        history = []
        for _ in range(TURNS):
            message = rng.choice(POOL)
            history.append(message)
            before = store.list_orders("demo_user")
            res = client.post("/api/chat", json={"session_id": session_id, "text": message})
            reply = res.json().get("reply", "") if res.status_code == 200 else f"HTTP {res.status_code}"
            for problem in _violations(session_id, message, reply, before):
                failures.append(f"{problem}  | after {history!r}")
    assert failures == [], "\n".join(failures[:5])
