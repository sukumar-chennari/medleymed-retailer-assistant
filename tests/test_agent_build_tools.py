"""Tests for the @tool-decorated closures inside _build_tools — each is a
LangChain StructuredTool wrapping a thin call into tools.py, directly
invokable via .invoke({...}) with no live agent or LLM needed at all,
the same insight behind every other test file added this week. The real
thing worth verifying here isn't tools.py's own logic (already covered
elsewhere) — it's that _build_tools wires each closure correctly,
especially start_order/reorder_last's session_id closure: that's the one
piece of wiring that's invisible to tools.py's own tests (which just take
session_id as an explicit argument) and would silently break the whole
order flow if it were ever wrong.

Uses the isolated_db fixture — several of these write real session/order
state through store.py, and must never touch the real demo data.
"""

import json

import pytest

from app import agent, store

pytestmark = pytest.mark.usefixtures("isolated_db")


def _tools_by_name(session_id: str) -> dict:
    return {t.name: t for t in agent._build_tools(session_id)}


class TestBuildToolsWiring:
    def test_lookup_symptom(self):
        tools_map = _tools_by_name("s1")
        result = json.loads(tools_map["lookup_symptom"].invoke({"symptom": "fever"}))
        assert result["matched"] is True
        assert result["category"] == "fever"

    def test_get_saved_address(self):
        store.save_address("demo_user", "1 Test Way")
        tools_map = _tools_by_name("s1")
        result = json.loads(tools_map["get_saved_address"].invoke({"user_id": "demo_user"}))
        assert result["address"] == "1 Test Way"

    def test_save_address(self):
        tools_map = _tools_by_name("s1")
        tools_map["save_address"].invoke({"user_id": "demo_user", "address": "2 Test Ave"})
        assert store.get_address("demo_user") == "2 Test Ave"

    def test_start_order_closes_over_the_right_session_id(self):
        # The real point of this test: session_id is closed over by
        # _build_tools, not passed by the caller — a wiring mistake here
        # (e.g. a swapped or hardcoded session_id) would silently break
        # every order for every user of this turn's tool set.
        tools_map = _tools_by_name("session-A")
        tools_map["start_order"].invoke({"product_id": "fev-001", "quantity": 1})
        assert store.get_pending_order("session-A") == {"product_id": "fev-001", "quantity": 1}
        assert store.get_pending_order("session-B") is None

    def test_reorder_last_closes_over_the_right_session_id(self):
        store.save_address("demo_user", "1 Test Way")
        store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        tools_map = _tools_by_name("session-A")
        tools_map["reorder_last"].invoke({"quantity": 1})
        assert store.get_pending_address_confirmation("session-A") is not None
        assert store.get_pending_address_confirmation("session-B") is None

    def test_check_order_status(self):
        store.save_address("demo_user", "1 Test Way")
        store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        tools_map = _tools_by_name("s1")
        result = json.loads(tools_map["check_order_status"].invoke({}))
        assert len(result["orders"]) == 1

    def test_cancel_order(self):
        store.save_address("demo_user", "1 Test Way")
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        tools_map = _tools_by_name("s1")
        result = json.loads(tools_map["cancel_order"].invoke({"order_id": order["order_id"]}))
        assert result["cancelled"] is True

    def test_lookup_medicine_info(self):
        tools_map = _tools_by_name("s1")
        result = json.loads(tools_map["lookup_medicine_info"].invoke({"query": "dosage for paracetamol 500mg"}))
        assert result["results"]
        assert result["results"][0]["product"] == "Paracetamol 500mg Tablets"

    def test_decline_out_of_scope(self):
        # Never actually reached in the real flow (wrap_tool_call always
        # short-circuits it first) — this just confirms the fallback text
        # itself, for completeness.
        tools_map = _tools_by_name("s1")
        assert "declined" in tools_map["decline_out_of_scope"].invoke({}).lower()
