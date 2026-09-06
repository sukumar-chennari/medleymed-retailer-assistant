"""Integration tests for the FastAPI routes in app/main.py that don't need
a live LLM call — /api/chat and the clarification-resolution path still go
through the real agent/Ollama and stay in the manual/live-testing category
(see DEMO_QA_PREP.md's evaluation section for why: an LLM reply is
nondeterministic enough that a naive assertion on exact text would be
flaky).

Uses the isolated_db fixture (see conftest.py) so these hit a fresh temp
database, not the real app.db backing the live demo.
"""

import pytest
from fastapi.testclient import TestClient

from app import main, store

pytestmark = pytest.mark.usefixtures("isolated_db")

client = TestClient(main.app)


class TestHealth:
    def test_health_ok(self):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


class TestDashboard:
    def test_dashboard_shape_with_no_orders(self):
        res = client.get("/api/dashboard")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "Demo User"
        assert data["address"] is None
        assert data["orders"] == []
        assert data["catalog_count"] == len(store.get_catalog())

    def test_dashboard_reflects_a_real_order(self):
        store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        res = client.get("/api/dashboard")
        data = res.json()
        assert len(data["orders"]) == 1
        assert data["orders"][0]["product_id"] == "fev-001"
        assert data["orders"][0]["status"] == "placed"


class TestCatalog:
    def test_catalog_returns_every_real_product(self):
        res = client.get("/api/catalog")
        assert res.status_code == 200
        products = res.json()
        assert len(products) == len(store.get_catalog())
        assert any(p["id"] == "fev-001" for p in products)


class TestMetrics:
    def test_metrics_empty_state(self):
        res = client.get("/api/metrics")
        assert res.status_code == 200
        data = res.json()
        assert data["avg_retrieval_confidence"] is None
        assert data["guardrail_total"] == 0
        assert data["recent_guardrail_events"] == []

    def test_metrics_reflects_a_logged_guardrail_event(self):
        store.log_metric_event("s1", "guardrail", "blocked_premature_order")
        res = client.get("/api/metrics")
        assert res.json()["guardrail_total"] == 1


class TestFeedback:
    def test_valid_feedback_is_accepted(self):
        res = client.post("/api/feedback", json={"session_id": "s1", "rating": "up", "reply_snippet": "test"})
        assert res.status_code == 200
        assert res.json() == {"ok": True}

    def test_feedback_is_reflected_in_metrics(self):
        client.post("/api/feedback", json={"session_id": "s1", "rating": "up"})
        res = client.get("/api/metrics")
        assert res.json()["feedback_positive"] == 1

    def test_invalid_rating_is_rejected(self):
        res = client.post("/api/feedback", json={"session_id": "s1", "rating": "sideways"})
        assert res.status_code == 400
