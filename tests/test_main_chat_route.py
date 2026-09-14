"""Integration tests for the /api/chat route's deterministic dispatch
branches in main.py's chat() — the ones that never call the LLM at all
(pending clarification/address-confirmation/email/order resolution, bare
product selection, the last-recommended-product affirmative shortcut).
This is the actual endpoint every one of those pending-state branches
lives in, previously only exercised end-to-end by the slow, LLM-dependent
app/conversation_eval.py script (or live curl testing) — these tests get
the same real dispatch logic under fast, reliable pytest by seeding
session state directly via store.set_pending_*/set_last_* rather than
driving a full multi-turn conversation through the LLM to reach it.

Uses the isolated_db fixture (see conftest.py) and forces
config.SMTP_CONFIGURED to False, same reasoning as test_tools_orders.py/
test_main_completion.py: this machine's real .env has live-demo SMTP
credentials, and a test suite must never risk a real send.

Anything that falls through to run_turn (a plain symptom description
with no pending state, a genuinely ambiguous reply) still needs the real
LLM and stays in conversation_eval.py / manual testing.
"""

import pytest
from fastapi.testclient import TestClient

from app import config, main, store

pytestmark = pytest.mark.usefixtures("isolated_db")

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _force_mock_email(monkeypatch):
    monkeypatch.setattr(config, "SMTP_CONFIGURED", False)


def _chat(session_id: str, text: str) -> str:
    res = client.post("/api/chat", json={"session_id": session_id, "text": text})
    res.raise_for_status()
    return res.json()["reply"]


class TestMissingInput:
    def test_no_text_and_no_image_is_rejected(self):
        res = client.post("/api/chat", json={"session_id": "s1", "text": ""})
        assert res.status_code == 400


class TestUnhandledErrorSafetyNet:
    def test_an_unexpected_exception_returns_a_friendly_200_not_a_500(self, monkeypatch):
        # Last-resort safety net (main.py's own comment on it): whatever
        # goes wrong, the user gets a real 200 with a friendly reply
        # instead of a raw 500 that skips this reply entirely.
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(store, "save_session_messages", _boom)
        store.set_pending_clarification("s1", "cough")  # any deterministic path reaches save_session_messages
        res = client.post("/api/chat", json={"session_id": "s1", "text": "dry"})
        assert res.status_code == 200
        assert res.json()["reply"] == main.GENERIC_FAILURE_REPLY


class TestPendingClarification:
    def test_a_clear_answer_resolves_deterministically(self):
        store.set_pending_clarification("s1", "cough")
        reply = _chat("s1", "dry")
        assert "Cough Suppressant" in reply
        assert store.get_pending_clarification("s1") is None


class TestPendingAddressConfirmation:
    def _seed(self, session_id: str):
        store.save_address("demo_user", "123 Main St")
        store.set_pending_address_confirmation(session_id, "fev-001", quantity=1)

    def test_affirmative_completes_the_order_with_the_address_on_file(self):
        self._seed("s1")
        reply = _chat("s1", "yes")
        assert "Order confirmed!" in reply
        assert "123 Main St" in reply
        assert store.get_pending_address_confirmation("s1") is None

    def test_a_new_address_replaces_the_one_on_file(self):
        self._seed("s1")
        reply = _chat("s1", "456 New Ave")
        assert "Order confirmed!" in reply
        assert "456 New Ave" in reply

    def test_wanting_a_different_address_asks_for_it_without_completing(self):
        self._seed("s1")
        reply = _chat("s1", "no, a different one please")
        assert "new shipping address" in reply.lower()
        # Still pending — the order isn't placed until the real address arrives.
        assert store.get_pending_address_confirmation("s1") is not None
        assert store.list_orders("demo_user") == []

    def test_ambiguous_reply_re_asks_the_compound_question(self):
        self._seed("s1")
        reply = _chat("s1", "hmm")
        assert "123 Main St" in reply
        assert "different address" in reply.lower()
        assert store.list_orders("demo_user") == []


class TestPendingEmail:
    def _seed(self, session_id: str) -> str:
        store.save_address("demo_user", "123 Main St")
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="123 Main St")
        store.set_pending_email(session_id, order["order_id"])
        return order["order_id"]

    def test_a_real_email_completes_it(self):
        self._seed("s1")
        reply = _chat("s1", "a@example.com")
        assert "sent the confirmation" in reply
        assert store.get_email("demo_user") == "a@example.com"
        assert store.get_pending_email("s1") is None

    def test_declining_leaves_the_order_confirmed_without_an_email(self):
        self._seed("s1")
        reply = _chat("s1", "no thanks")
        assert "already confirmed" in reply.lower()
        assert store.get_pending_email("s1") is None
        assert store.get_email("demo_user") is None


class TestPendingOrderNoAddressYet:
    def _seed(self, session_id: str):
        store.set_pending_order(session_id, "fev-001", quantity=1)

    def test_supplying_an_address_completes_the_order(self):
        self._seed("s1")
        reply = _chat("s1", "123 First Time Rd")
        assert "Order confirmed!" in reply
        assert "123 First Time Rd" in reply

    def test_declining_cancels_the_pending_order_without_placing_it(self):
        self._seed("s1")
        reply = _chat("s1", "no, forget it")
        assert "won't place that order" in reply.lower()
        assert store.list_orders("demo_user") == []
        assert store.get_pending_order("s1") is None


class TestBareSelectionAndLastRecommended:
    def test_bare_numeric_selection_starts_the_real_order(self):
        products = [
            {"id": "fev-001", "name": "Paracetamol 500mg Tablets", "description": "x"},
            {"id": "fev-002", "name": "Paracetamol Extra Strength 650mg", "description": "y"},
        ]
        store.set_last_products("s1", products)
        reply = _chat("s1", "2")
        # No address on file yet in this isolated DB — start_order defers.
        assert "shipping address" in reply.lower()
        assert store.get_pending_order("s1") == {"product_id": "fev-002", "quantity": 1}

    def test_bare_affirmative_after_a_single_recommendation_starts_the_real_order(self):
        store.set_last_recommended_product("s1", "fev-001")
        reply = _chat("s1", "yes")
        assert "shipping address" in reply.lower()
        assert store.get_pending_order("s1") == {"product_id": "fev-001", "quantity": 1}
        assert store.get_last_recommended_product("s1") is None
