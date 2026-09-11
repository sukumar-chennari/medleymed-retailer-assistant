"""Tests for the order-placement flow in app/tools.py: place_order,
start_order, reorder_last, complete_confirmed_order, check_order_status,
cancel_order, and send_confirmation_email. All deterministic and DB-only
(no LLM call anywhere in this flow) — this is the business logic behind
every order-related chat exchange that's otherwise only ever been
exercised through live curl testing.

Uses the isolated_db fixture (see conftest.py). Also forces
config.SMTP_CONFIGURED to False for every test in this file regardless of
the real environment: this machine's actual .env has real SMTP
credentials configured for the live demo, and a test suite must never
risk sending a real email — forcing the mock path keeps
send_confirmation_email's behavior deterministic here no matter what the
real environment looks like.
"""

import json
import smtplib
from unittest import mock

import pytest

from app import config, store, tools

pytestmark = pytest.mark.usefixtures("isolated_db")


@pytest.fixture(autouse=True)
def _force_mock_email(monkeypatch):
    monkeypatch.setattr(config, "SMTP_CONFIGURED", False)


class TestSendConfirmationEmail:
    def test_mock_mode_reports_sent_without_a_real_smtp_call(self):
        result = json.loads(tools.send_confirmation_email("a@example.com", "order summary"))
        assert result == {"sent": True, "mode": "mock"}


class TestSendConfirmationEmailRealSmtpPath:
    """Exercises the branch send_confirmation_email takes when
    config.SMTP_CONFIGURED is True — smtplib.SMTP itself is mocked out
    entirely (never a real socket/network call), since a test suite must
    never risk contacting a real mail server, let alone with this
    machine's real live-demo credentials."""

    def _configure_smtp(self, monkeypatch):
        monkeypatch.setattr(config, "SMTP_CONFIGURED", True)
        monkeypatch.setattr(config, "SMTP_HOST", "smtp.example.com")
        monkeypatch.setattr(config, "SMTP_PORT", "587")
        monkeypatch.setattr(config, "SMTP_USER", "sender@example.com")
        monkeypatch.setattr(config, "SMTP_PASS", "secret")
        monkeypatch.setattr(config, "SMTP_FROM_NAME", "MedleyMed Orders")

    def test_successful_send(self, monkeypatch):
        self._configure_smtp(monkeypatch)
        fake_smtp = mock.MagicMock()
        fake_smtp.__enter__ = mock.Mock(return_value=fake_smtp)
        fake_smtp.__exit__ = mock.Mock(return_value=False)

        with mock.patch("app.tools.smtplib.SMTP", return_value=fake_smtp) as mock_smtp_cls:
            result = json.loads(tools.send_confirmation_email("a@example.com", "order summary"))

        assert result == {"sent": True, "mode": "smtp"}
        mock_smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=10)
        fake_smtp.starttls.assert_called_once()
        fake_smtp.login.assert_called_once_with("sender@example.com", "secret")
        fake_smtp.send_message.assert_called_once()

    def test_smtp_failure_is_caught_and_reported_not_raised(self, monkeypatch):
        # An order should never fail just because email delivery did — see
        # send_confirmation_email's own comment on this.
        self._configure_smtp(monkeypatch)
        with mock.patch("app.tools.smtplib.SMTP", side_effect=smtplib.SMTPConnectError(421, "connection refused")):
            result = json.loads(tools.send_confirmation_email("a@example.com", "order summary"))

        assert result["sent"] is False
        assert result["mode"] == "error"
        assert "error" in result


class TestSavedAddress:
    def test_no_address_saved_yet(self):
        result = json.loads(tools.get_saved_address("demo_user"))
        assert result["address"] is None

    def test_save_then_get_round_trip(self):
        save_result = json.loads(tools.save_address("demo_user", "1 Test Way"))
        assert save_result == {"saved": True, "address": "1 Test Way"}
        get_result = json.loads(tools.get_saved_address("demo_user"))
        assert get_result["address"] == "1 Test Way"


class TestPlaceOrder:
    def test_unknown_product_returns_an_error(self):
        result = json.loads(tools.place_order("not-a-real-id"))
        assert "error" in result

    def test_no_address_on_file_returns_an_error(self):
        result = json.loads(tools.place_order("fev-001"))
        assert "error" in result
        assert "No address on file" in result["error"]

    def test_valid_order_is_actually_created(self):
        store.save_address("demo_user", "1 Test Way")
        result = json.loads(tools.place_order("fev-001"))
        assert result["order_id"] == "ord-0001"
        assert result["product_id"] == "fev-001"
        assert store.get_order("ord-0001") is not None

    def test_quantity_over_the_limit_is_clamped_and_noted(self):
        store.save_address("demo_user", "1 Test Way")
        result = json.loads(tools.place_order("fev-001", quantity=tools.MAX_QUANTITY_PER_ORDER + 5))
        assert result["quantity"] == tools.MAX_QUANTITY_PER_ORDER
        assert "quantity_clamped" in result


class TestStartOrder:
    def test_unknown_product_returns_an_error(self):
        result = json.loads(tools.start_order("not-a-real-id", "s1"))
        assert "error" in result

    def test_no_address_defers_and_remembers_the_pending_order(self):
        result = json.loads(tools.start_order("fev-001", "s1"))
        assert result["order_placed"] is False
        assert "No address on file" in result["message"]
        assert store.get_pending_order("s1") == {"product_id": "fev-001", "quantity": 1}

    def test_address_on_file_defers_to_confirm_it(self):
        store.save_address("demo_user", "1 Test Way")
        result = json.loads(tools.start_order("fev-001", "s1"))
        assert result["order_placed"] is False
        assert result["needs_address_confirmation"] is True
        assert result["address_on_file"] == "1 Test Way"
        assert store.get_pending_address_confirmation("s1") == {"product_id": "fev-001", "quantity": 1}

    def test_never_creates_a_real_order_by_itself(self):
        store.save_address("demo_user", "1 Test Way")
        tools.start_order("fev-001", "s1")
        assert store.list_orders("demo_user") == []


class TestReorderLast:
    def test_no_previous_orders_returns_an_error(self):
        result = json.loads(tools.reorder_last("s1"))
        assert "error" in result

    def test_resolves_to_the_most_recent_orders_product(self):
        store.save_address("demo_user", "1 Test Way")
        store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        store.create_order(user_id="demo_user", product_id="col-001", address="1 Test Way")
        result = json.loads(tools.reorder_last("s1"))
        # Delegates to start_order, which defers to address confirmation
        # for col-001 (the most recently created order) since an address
        # is already on file.
        assert result["needs_address_confirmation"] is True
        assert store.get_pending_address_confirmation("s1")["product_id"] == "col-001"

    def test_includes_a_previously_cancelled_order_as_reorderable(self):
        store.save_address("demo_user", "1 Test Way")
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        store.cancel_order(order["order_id"], "demo_user")
        result = json.loads(tools.reorder_last("s1"))
        assert store.get_pending_address_confirmation("s1")["product_id"] == "fev-001"


class TestCompleteConfirmedOrder:
    def test_unknown_product_error_passes_through(self):
        result = json.loads(tools.complete_confirmed_order("s1", "not-a-real-id", 1))
        assert "error" in result

    def test_no_email_on_file_marks_order_placed_and_defers_email(self):
        store.save_address("demo_user", "1 Test Way")
        result = json.loads(tools.complete_confirmed_order("s1", "fev-001", 1))
        assert result["order_placed"] is True
        assert result["email_needed"] is True
        assert result["email_sent"] is False
        assert store.get_pending_email("s1") == result["order_id"]

    def test_email_on_file_sends_a_mock_confirmation(self):
        store.save_address("demo_user", "1 Test Way")
        store.save_email("demo_user", "a@example.com")
        result = json.loads(tools.complete_confirmed_order("s1", "fev-001", 1))
        assert result["order_placed"] is True
        assert result["email_sent"] is True
        assert store.get_pending_email("s1") is None


class TestCheckOrderStatus:
    def test_no_orders_returns_a_friendly_message(self):
        result = json.loads(tools.check_order_status())
        assert result["orders"] == []
        assert "message" in result

    def test_returns_the_real_orders(self):
        store.save_address("demo_user", "1 Test Way")
        store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        result = json.loads(tools.check_order_status())
        assert len(result["orders"]) == 1
        assert result["orders"][0]["product_id"] == "fev-001"


class TestCancelOrder:
    def test_no_orders_at_all_returns_an_error(self):
        result = json.loads(tools.cancel_order())
        assert "error" in result

    def test_empty_order_id_cancels_the_most_recent_active_order(self):
        store.save_address("demo_user", "1 Test Way")
        store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        second = store.create_order(user_id="demo_user", product_id="col-001", address="1 Test Way")
        result = json.loads(tools.cancel_order())
        assert result["order_id"] == second["order_id"]
        assert result["cancelled"] is True

    def test_skips_an_already_cancelled_order_when_picking_the_most_recent_active_one(self):
        store.save_address("demo_user", "1 Test Way")
        first = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        second = store.create_order(user_id="demo_user", product_id="col-001", address="1 Test Way")
        store.cancel_order(second["order_id"], "demo_user")
        result = json.loads(tools.cancel_order())
        assert result["order_id"] == first["order_id"]

    def test_explicit_order_id_cancels_that_specific_order(self):
        store.save_address("demo_user", "1 Test Way")
        first = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        store.create_order(user_id="demo_user", product_id="col-001", address="1 Test Way")
        result = json.loads(tools.cancel_order(first["order_id"]))
        assert result["order_id"] == first["order_id"]
