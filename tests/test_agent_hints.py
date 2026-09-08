"""Tests for the deterministic message-building/session-tracking/logging
helpers in app/agent.py that aren't part of the clarifying-question state
machine covered by test_agent.py: _inject_catalog_hint (grounds a
plain-text or image-description mention in real catalog data before the
model ever sees it), _established_category and _remember_products (back
the "don't trust the model to remember what category/products it just
showed" guards documented throughout agent.py), and _log_guard/
_log_tool_call (the two functions that turn "we have guardrails" from a
doc claim into the real, queryable metrics_events rows the dashboard
reads).

Uses the isolated_db fixture (see conftest.py) since these read/write
session state and metrics through store.py.
"""

import pytest

from app import agent, store

pytestmark = pytest.mark.usefixtures("isolated_db")


class TestInjectCatalogHint:
    def test_info_question_appends_nothing(self):
        # Real bug this guards: the injected hint used to bias the model
        # toward "present as a recommendation" even for a factual dosage
        # question, steering it away from ever calling lookup_medicine_info.
        parts = ["dosage for paracetamol 500mg"]
        agent._inject_catalog_hint(parts, "dosage for paracetamol 500mg", "s1")
        assert parts == ["dosage for paracetamol 500mg"]

    def test_ambiguous_symptom_appends_a_clarify_hint(self):
        parts = ["I have a cough"]
        agent._inject_catalog_hint(parts, "I have a cough", "s1")
        assert len(parts) == 2
        assert parts[1].startswith("[Clarify:")
        assert "dry" in parts[1].lower()

    def test_qualified_symptom_appends_real_catalog_data(self):
        parts = ["my child has a sore throat"]
        agent._inject_catalog_hint(parts, "my child has a sore throat", "s1")
        assert len(parts) == 2
        assert "cold-related" in parts[1]
        assert "col-" in parts[1]  # a real product_id, not an invented one

    def test_unrelated_message_appends_nothing(self):
        parts = ["what's the weather today"]
        agent._inject_catalog_hint(parts, "what's the weather today", "s1")
        assert parts == ["what's the weather today"]


class TestEstablishedCategory:
    def test_no_prior_products_means_no_established_category(self):
        assert agent._established_category("s1") is None

    def test_a_fever_product_shown_establishes_fever(self):
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        assert agent._established_category("s1") == "fever"

    def test_a_cold_product_shown_establishes_cold(self):
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "col-001"}]}')
        assert agent._established_category("s1") == "cold"

    def test_sessions_are_independent(self):
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        agent._remember_products("s2", '{"matched": true, "products": [{"id": "col-001"}]}')
        assert agent._established_category("s1") == "fever"
        assert agent._established_category("s2") == "cold"


class TestRememberProducts:
    def test_stores_products_from_a_matched_result(self):
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        assert store.get_last_products("s1") == [{"id": "fev-001"}]

    def test_does_not_store_anything_for_an_unmatched_result(self):
        agent._remember_products("s1", '{"matched": false, "message": "no match"}')
        assert store.get_last_products("s1") is None

    def test_invalid_json_does_not_raise(self):
        agent._remember_products("s1", "not valid json")  # must not raise


class TestRenderProducts:
    def test_single_product_asks_to_order_and_remembers_it_as_recommended(self):
        product = store.find_product("fev-001")
        reply = agent._render_products([product], "s1")
        assert "Would you like to order this?" in reply
        assert product["name"] in reply
        assert store.get_last_recommended_product("s1") == "fev-001"
        assert store.get_last_products("s1") == [product]

    def test_multiple_products_lists_them_without_a_single_recommendation(self):
        products = [store.find_product("fev-001"), store.find_product("fev-002")]
        reply = agent._render_products(products, "s1")
        assert "Which one would you like to try?" in reply
        assert "fev-001" in reply and "fev-002" in reply
        assert store.get_last_recommended_product("s1") is None
        assert store.get_last_products("s1") == products

    def test_every_reply_includes_the_disclaimer(self):
        from app import guardrails

        reply = agent._render_products([store.find_product("fev-001")], "s1")
        assert guardrails.DISCLAIMER in reply


class TestLogGuardAndLogToolCall:
    def test_log_guard_writes_a_queryable_guardrail_event(self):
        agent._log_guard("s1", "blocked_premature_order", "some detail")
        summary = store.get_metrics_summary()
        counts = {row["name"]: row["count"] for row in summary["guardrail_counts"]}
        assert counts["blocked_premature_order"] == 1
        assert summary["recent_guardrail_events"][0]["detail"] == "some detail"

    def test_log_guard_without_detail_still_logs(self):
        agent._log_guard("s1", "blocked_unconfirmed_cancel")
        summary = store.get_metrics_summary()
        assert summary["guardrail_total"] == 1

    def test_log_tool_call_writes_a_queryable_tool_call_event(self):
        agent._log_tool_call("s1", "start_order", {"product_id": "fev-001"}, "{}")
        summary = store.get_metrics_summary()
        counts = {row["name"]: row["count"] for row in summary["tool_call_counts"]}
        assert counts["start_order"] == 1
