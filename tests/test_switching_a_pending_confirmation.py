"""Real bug (found Nov 17): with "ship to the address on file, or a different one?" open,
"order ibuprofen" / "actually paracetamol 650mg instead" went to the model and left the OLD
confirmation pending, so the customer's next "yes" shipped the product they had just said
they did not want."""

import pytest
from fastapi.testclient import TestClient

from app import config, main, store

pytestmark = pytest.mark.usefixtures("isolated_db")
client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(config, "SMTP_CONFIGURED", False)
    monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))


def _chat(session_id, text):
    res = client.post("/api/chat", json={"session_id": session_id, "text": text})
    res.raise_for_status()
    return res.json()["reply"]


def _waiting_on_paracetamol_500():
    store.save_address("demo_user", "12 Baker Street, London")
    reply = _chat("s1", "order paracetamol 500mg")
    assert "We have this address on file" in reply
    assert store.get_pending_address_confirmation("s1")["product_id"] == "fev-001"


@pytest.mark.parametrize("text, product_id, quantity", [
    ("order 2 ibuprofen", "fev-003", 2),
    ("actually paracetamol 650mg instead", "fev-002", 1),
    ("I want cetirizine", "col-001", 1),
    ("order paracetamol 650", "fev-002", 1),
])
def test_a_new_named_order_replaces_the_waiting_confirmation(text, product_id, quantity):
    _waiting_on_paracetamol_500()
    reply = _chat("s1", text)
    assert "We have this address on file" in reply
    assert store.get_pending_address_confirmation("s1") == {"product_id": product_id, "quantity": quantity}
    reply = _chat("s1", "yes")
    assert "Order confirmed!" in reply
    orders = store.list_orders("demo_user")
    assert [(o["product_id"], o["quantity"]) for o in orders] == [(product_id, quantity)]


def test_an_unrelated_message_still_leaves_the_confirmation_open():
    _waiting_on_paracetamol_500()
    assert _chat("s1", "who is narendra modi") == "from the model"
    assert store.get_pending_address_confirmation("s1")["product_id"] == "fev-001"
    assert "Order confirmed!" in _chat("s1", "yes")
    assert store.list_orders("demo_user")[0]["product_id"] == "fev-001"


def test_switching_to_ibuprofen_for_a_child_is_refused_and_nothing_is_ordered():
    # also covers the age gloss being skipped while an order is pending
    _waiting_on_paracetamol_500()
    reply = _chat("s1", "order ibuprofen for my 5 year old")
    assert "adult dose only" in reply
    _chat("s1", "yes")
    assert all(o["product_id"] != "fev-003" for o in store.list_orders("demo_user"))
