"""Tests for the deterministic message-building/session-tracking helpers in
app/agent.py that aren't part of the clarifying-question state machine
covered by test_agent.py: _inject_catalog_hint (grounds a plain-text or
image-description mention in real catalog data before the model ever sees
it), _established_category, and _remember_products (both back the
"don't trust the model to remember what category/products it just showed"
guards documented throughout agent.py).

Uses the isolated_db fixture (see conftest.py) since these read/write
session state through store.py.
"""

import pytest

from app import agent

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
        from app import store

        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        assert store.get_last_products("s1") == [{"id": "fev-001"}]

    def test_does_not_store_anything_for_an_unmatched_result(self):
        from app import store

        agent._remember_products("s1", '{"matched": false, "message": "no match"}')
        assert store.get_last_products("s1") is None

    def test_invalid_json_does_not_raise(self):
        agent._remember_products("s1", "not valid json")  # must not raise
