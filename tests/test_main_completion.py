"""Tests for the deterministic order-completion functions in app/main.py:
_complete_pending_order (finishes an order once the user's next message
supplies the address), _complete_confirmed_order (finishes it once the
user confirms the address already on file), _reply_for_start_order_result,
and _complete_pending_email. These exist specifically so the LLM never has
to reliably chain save_address + place_order + send_confirmation_email
itself across turns — see their own docstrings.

Uses the isolated_db fixture (see conftest.py). Also forces
config.SMTP_CONFIGURED to False for every test here, same reasoning as
test_tools_orders.py: this machine's real .env has real SMTP credentials
configured for the live demo, so a test suite must never risk sending a
real email.
"""

import pytest

from app import config, main, store

pytestmark = pytest.mark.usefixtures("isolated_db")


@pytest.fixture(autouse=True)
def _force_mock_email(monkeypatch):
    monkeypatch.setattr(config, "SMTP_CONFIGURED", False)


class TestCompletePendingOrder:
    def test_saves_the_address_and_places_a_real_order(self):
        reply = main._complete_pending_order("s1", {"product_id": "fev-001", "quantity": 1}, "123 Main St")
        assert "Order confirmed!" in reply
        assert store.get_address("demo_user") == "123 Main St"
        orders = store.list_orders("demo_user")
        assert len(orders) == 1
        assert orders[0]["product_id"] == "fev-001"

    def test_strips_ship_to_filler_before_saving_the_address(self):
        main._complete_pending_order("s1", {"product_id": "fev-001", "quantity": 1}, "please ship to 123 Main St")
        assert store.get_address("demo_user") == "123 Main St"

    def test_an_email_included_in_the_address_text_is_extracted_and_saved(self):
        main._complete_pending_order("s1", {"product_id": "fev-001", "quantity": 1}, "123 Main St a@example.com")
        assert store.get_address("demo_user") == "123 Main St"
        assert store.get_email("demo_user") == "a@example.com"

    def test_no_email_on_file_defers_asking_for_one(self):
        reply = main._complete_pending_order("s1", {"product_id": "fev-001", "quantity": 1}, "123 Main St")
        assert "reply with your email" in reply.lower() or "email" in reply.lower()
        orders = store.list_orders("demo_user")
        assert store.get_pending_email("s1") == orders[0]["order_id"]

    def test_unknown_product_returns_an_apologetic_error(self):
        reply = main._complete_pending_order("s1", {"product_id": "not-a-real-id", "quantity": 1}, "123 Main St")
        assert "Sorry" in reply
        assert store.list_orders("demo_user") == []


class TestCompleteConfirmedOrder:
    def test_places_the_order_using_the_address_already_on_file(self):
        store.save_address("demo_user", "123 Main St")
        reply = main._complete_confirmed_order("s1", {"product_id": "fev-001", "quantity": 1})
        assert "Order confirmed!" in reply
        assert "123 Main St" in reply
        assert len(store.list_orders("demo_user")) == 1

    def test_unknown_product_returns_an_apologetic_error(self):
        store.save_address("demo_user", "123 Main St")
        reply = main._complete_confirmed_order("s1", {"product_id": "not-a-real-id", "quantity": 1})
        assert "Sorry" in reply
        assert store.list_orders("demo_user") == []


class TestReplyForStartOrderResult:
    def test_error_is_relayed_apologetically(self):
        reply = main._reply_for_start_order_result({"error": "Unknown product_id 'x'"})
        assert reply.startswith("Sorry,")

    def test_deferred_order_asks_for_or_confirms_an_address(self):
        order = {"order_placed": False, "needs_address_confirmation": False}
        reply = main._reply_for_start_order_result(order)
        assert "shipping address" in reply.lower()

    def test_completed_order_returns_the_real_confirmation(self):
        order = {
            "order_placed": True, "order_id": "ord-0001", "product_name": "Paracetamol 500mg Tablets",
            "quantity": 1, "total_price_usd": 4.99, "address": "123 Main St", "email_sent": False,
        }
        reply = main._reply_for_start_order_result(order)
        assert "Order confirmed!" in reply
        assert "ord-0001" in reply


class TestCompletePendingEmail:
    def test_saves_the_email_and_sends_a_confirmation_for_a_real_order(self):
        store.save_address("demo_user", "123 Main St")
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="123 Main St")
        reply = main._complete_pending_email("a@example.com", order["order_id"])
        assert "sent the confirmation" in reply
        assert store.get_email("demo_user") == "a@example.com"

    def test_saves_the_email_even_if_the_order_cannot_be_found(self):
        reply = main._complete_pending_email("a@example.com", "ord-9999")
        assert "couldn't find that earlier order" in reply
        assert store.get_email("demo_user") == "a@example.com"
