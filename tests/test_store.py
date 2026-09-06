"""Regression tests for app/store.py, running against an isolated temp DB
(see conftest.py's isolated_db fixture) rather than the real app.db that
backs the live demo.

Covers the parts of the persistence layer that have real behavioral
guarantees the rest of the app depends on: order creation/ordering ids,
cancellation (status flip, not deletion — see the migration comment in
store.py), address/session-state round trips, and the metrics aggregation
that the dashboard reads.
"""

import json

import pytest

from app import store

pytestmark = pytest.mark.usefixtures("isolated_db")


class TestOrders:
    def test_create_order_appears_in_list_orders(self):
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        orders = store.list_orders("demo_user")
        assert len(orders) == 1
        assert orders[0]["order_id"] == order["order_id"]
        assert orders[0]["product_id"] == "fev-001"
        assert orders[0]["status"] == "placed"

    def test_order_ids_increment(self):
        first = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        second = store.create_order(user_id="demo_user", product_id="fev-002", address="1 Test Way")
        assert first["order_id"] == "ord-0001"
        assert second["order_id"] == "ord-0002"

    def test_total_price_reflects_quantity(self):
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way", quantity=2)
        assert order["total_price_usd"] == round(order["price_usd"] * 2, 2)

    def test_unknown_product_raises(self):
        with pytest.raises(ValueError):
            store.create_order(user_id="demo_user", product_id="not-a-real-id", address="1 Test Way")


class TestCancelOrder:
    def test_cancel_marks_status_without_deleting(self):
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        result = store.cancel_order(order["order_id"], "demo_user")
        assert result["cancelled"] is True
        assert result["status"] == "cancelled"
        # Still present in history, not deleted.
        assert store.get_order(order["order_id"])["status"] == "cancelled"

    def test_cancel_unknown_order_returns_error(self):
        result = store.cancel_order("ord-9999", "demo_user")
        assert "error" in result

    def test_cancel_wrong_user_returns_error(self):
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        result = store.cancel_order(order["order_id"], "someone_else")
        assert "error" in result
        assert store.get_order(order["order_id"])["status"] == "placed"

    def test_cancelling_twice_returns_error_the_second_time(self):
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        store.cancel_order(order["order_id"], "demo_user")
        result = store.cancel_order(order["order_id"], "demo_user")
        assert "error" in result


class TestAddressAndEmail:
    def test_save_and_get_address(self):
        assert store.get_address("demo_user") is None
        store.save_address("demo_user", "42 New St")
        assert store.get_address("demo_user") == "42 New St"

    def test_save_and_get_email(self):
        assert store.get_email("demo_user") is None
        store.save_email("demo_user", "a@example.com")
        assert store.get_email("demo_user") == "a@example.com"


class TestSessionState:
    def test_pending_order_round_trip(self):
        session_id = "s1"
        assert store.get_pending_order(session_id) is None
        store.set_pending_order(session_id, "fev-001", quantity=2)
        assert store.get_pending_order(session_id) == {"product_id": "fev-001", "quantity": 2}
        store.clear_pending_order(session_id)
        assert store.get_pending_order(session_id) is None

    def test_pending_clarification_round_trip(self):
        session_id = "s2"
        store.set_pending_clarification(session_id, "cough:child")
        assert store.get_pending_clarification(session_id) == "cough:child"
        store.clear_pending_clarification(session_id)
        assert store.get_pending_clarification(session_id) is None

    def test_sessions_are_independent(self):
        store.set_pending_order("s3", "fev-001")
        store.set_pending_order("s4", "fev-002")
        assert store.get_pending_order("s3")["product_id"] == "fev-001"
        assert store.get_pending_order("s4")["product_id"] == "fev-002"


class TestMetrics:
    def test_retrieval_average_from_logged_scores(self):
        store.log_metric_event("s1", "retrieval", "lookup_medicine_info", value=0.8)
        store.log_metric_event("s1", "retrieval", "lookup_medicine_info", value=0.9)
        summary = store.get_metrics_summary()
        assert summary["avg_retrieval_confidence"] == 0.85
        assert summary["retrieval_sample_count"] == 2

    def test_guardrail_counts_grouped_by_name(self):
        store.log_metric_event("s1", "guardrail", "blocked_premature_order")
        store.log_metric_event("s1", "guardrail", "blocked_premature_order")
        store.log_metric_event("s1", "guardrail", "blocked_unconfirmed_cancel")
        summary = store.get_metrics_summary()
        counts = {row["name"]: row["count"] for row in summary["guardrail_counts"]}
        assert counts["blocked_premature_order"] == 2
        assert counts["blocked_unconfirmed_cancel"] == 1
        assert summary["guardrail_total"] == 3

    def test_feedback_positive_rate(self):
        store.log_metric_event("s1", "feedback", "up")
        store.log_metric_event("s1", "feedback", "up")
        store.log_metric_event("s1", "feedback", "down")
        summary = store.get_metrics_summary()
        assert summary["feedback_total"] == 3
        assert summary["feedback_positive_rate"] == round(2 / 3, 3)

    def test_no_events_yields_none_averages_not_errors(self):
        summary = store.get_metrics_summary()
        assert summary["avg_retrieval_confidence"] is None
        assert summary["feedback_positive_rate"] is None
        assert summary["guardrail_counts"] == []

    def test_recent_guardrail_events_include_detail_and_are_newest_first(self):
        store.log_metric_event("s1", "guardrail", "first_event", detail="oldest")
        store.log_metric_event("s1", "guardrail", "second_event", detail="newest")
        events = store.get_metrics_summary()["recent_guardrail_events"]
        assert events[0]["name"] == "second_event"
        assert events[0]["detail"] == "newest"
        assert events[1]["name"] == "first_event"
