"""Tests for run_turn itself — not just its constituent helper functions
(deterministic_pleasantry_reply, _needs_clarification,
_resolve_prequalified_clarification, individually covered elsewhere) —
for the three early-return paths at the top of the function that never
reach the LLM at all: a pleasantry, a message that needs a clarifying
question, and a message that arrives pre-qualified. Each returns before
_build_user_text/the agent graph are ever touched, so these are exactly
as cheap to test directly as the middleware hooks were.

Anything that falls through past these three checks needs the real chat
model and stays in app/conversation_eval.py / manual testing.

Uses the isolated_db fixture since these read/write session state
through store.py.
"""

import pytest

from app import agent, guardrails, store

pytestmark = pytest.mark.usefixtures("isolated_db")


class TestRunTurnPleasantryShortCircuit:
    def test_a_greeting_returns_immediately_with_no_pending_state(self):
        reply, messages = agent.run_turn([], "hi", "s1")
        assert reply == guardrails.GREETING_REPLY
        assert messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": guardrails.GREETING_REPLY},
        ]
        assert store.get_pending_clarification("s1") is None

    def test_appends_onto_existing_message_history(self):
        history = [{"role": "user", "content": "earlier turn"}]
        reply, messages = agent.run_turn(history, "thanks", "s1")
        assert reply == guardrails.BYE_REPLY
        assert messages[0] == {"role": "user", "content": "earlier turn"}
        assert messages[-1] == {"role": "assistant", "content": guardrails.BYE_REPLY}


class TestRunTurnClarificationShortCircuit:
    def test_an_ambiguous_symptom_asks_the_question_and_sets_pending_state(self):
        reply, messages = agent.run_turn([], "I have a cough", "s1")
        assert "dry" in reply.lower()
        assert messages[-1] == {"role": "assistant", "content": reply}
        assert store.get_pending_clarification("s1") == "cough"

    def test_an_already_qualified_symptom_does_not_set_pending_state(self):
        # Falls through this check (nothing to ask) into the prequalified
        # one below instead.
        agent.run_turn([], "my baby has a fever", "s1")
        assert store.get_pending_clarification("s1") is None


class TestRunTurnPrequalifiedShortCircuit:
    def test_a_prequalified_symptom_resolves_directly_to_a_real_product(self):
        reply, messages = agent.run_turn([], "my baby has a fever", "s1")
        assert "Children's Paracetamol Syrup" in reply
        assert messages[-1] == {"role": "assistant", "content": reply}

    def test_a_prequalified_child_wet_cough_declines_safely(self):
        # Real, previously-shipped bug (see agent.py's _AGE_WORDS comment):
        # this used to recommend the adult expectorant to a child.
        reply, _ = agent.run_turn([], "my child has a wet cough", "s1")
        assert "pharmacist" in reply.lower()
        assert "guaifenesin" not in reply.lower()
