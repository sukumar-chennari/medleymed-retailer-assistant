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


class _ReachedTheAgent(Exception):
    """Raised by a stub standing in for the LLM/agent graph, so a test can
    prove run_turn got PAST its deterministic short-circuits without ever
    needing a real model."""


def _read_fails(monkeypatch):
    monkeypatch.setattr(agent, "describe_image", lambda image_b64, media_type: agent._IMAGE_UNREADABLE)


def _agent_must_not_run(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the LLM/agent graph must not be reached for this turn")

    monkeypatch.setattr(agent, "create_agent", boom)
    monkeypatch.setattr(agent, "_build_user_text", boom)


class TestRunTurnUnreadablePhotoShortCircuit:
    """Real bug: a photo that failed to read (Gemini error, no API key, empty
    response) was still treated as grounding for lookup_symptom, and with no
    text the only guard left forced OUT_OF_SCOPE_REPLY ("I can't help with
    that here") — wrong for a blurry photo. A photo-only message that can't
    be read now gets a deterministic "couldn't read that photo" reply, like
    pleasantries and clarifying questions do, with no LLM turn at all."""

    def test_a_photo_only_message_that_cannot_be_read_gets_the_deterministic_reply(self, monkeypatch):
        _read_fails(monkeypatch)
        _agent_must_not_run(monkeypatch)
        reply, messages = agent.run_turn([], "", "s1", image_b64="ZmFrZQ==", image_media_type="image/png")
        assert reply == agent.IMAGE_UNREADABLE_REPLY
        assert "type the medicine name" in reply
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[-1] == {"role": "assistant", "content": agent.IMAGE_UNREADABLE_REPLY}
        assert store.get_last_products("s1") is None  # nothing was looked up

    def test_whitespace_only_text_counts_as_no_text(self, monkeypatch):
        _read_fails(monkeypatch)
        _agent_must_not_run(monkeypatch)
        reply, _ = agent.run_turn([], "   ", "s1", image_b64="ZmFrZQ==")
        assert reply == agent.IMAGE_UNREADABLE_REPLY

    def test_no_vision_api_configured_is_also_unreadable(self, monkeypatch):
        monkeypatch.setattr(agent, "_gemini_client", None)
        _agent_must_not_run(monkeypatch)
        reply, _ = agent.run_turn([], "", "s1", image_b64="ZmFrZQ==")
        assert reply == agent.IMAGE_UNREADABLE_REPLY

    def test_a_readable_photo_is_not_short_circuited(self, monkeypatch):
        monkeypatch.setattr(agent, "describe_image", lambda image_b64, media_type: "Paracetamol 500mg Tablets box")

        def reached(*args, **kwargs):
            raise _ReachedTheAgent

        monkeypatch.setattr(agent, "create_agent", reached)
        with pytest.raises(_ReachedTheAgent):
            agent.run_turn([], "", "s1", image_b64="ZmFrZQ==", image_media_type="image/png")


class TestRunTurnPhotoGrounding:
    """The middleware is built from run_turn's own grounding decision, so
    capture what it's given instead of running a model."""

    def _capture_middleware_kwargs(self, monkeypatch, describe_result: str, user_text: str) -> dict:
        monkeypatch.setattr(agent, "describe_image", lambda image_b64, media_type: describe_result)
        captured: dict = {}

        def fake_middleware(**kwargs):
            captured.update(kwargs)
            raise _ReachedTheAgent

        monkeypatch.setattr(agent, "_GuardrailMiddleware", fake_middleware)
        with pytest.raises(_ReachedTheAgent):
            agent.run_turn([], user_text, "s1", image_b64="ZmFrZQ==", image_media_type="image/png")
        return captured

    def test_a_photo_that_could_not_be_read_does_not_ground_a_lookup(self, monkeypatch):
        # Real bug: this used to be bool(image_b64), so a failed read still
        # licensed lookup_symptom and bypassed blocked_ungrounded_lookup.
        captured = self._capture_middleware_kwargs(monkeypatch, agent._IMAGE_UNREADABLE, "what is this?")
        assert captured["image_read"] is False
        assert captured["symptom_lookup_grounded"] is False

    def test_a_photo_that_was_read_still_grounds_a_lookup(self, monkeypatch):
        captured = self._capture_middleware_kwargs(monkeypatch, "Paracetamol 500mg Tablets box", "what is this?")
        assert captured["image_read"] is True
        assert captured["symptom_lookup_grounded"] is True

    def test_text_that_classifies_still_grounds_even_if_the_photo_failed(self, monkeypatch):
        captured = self._capture_middleware_kwargs(monkeypatch, agent._IMAGE_UNREADABLE, "I have a fever, here's the box")
        assert captured["image_read"] is False
        assert captured["symptom_lookup_grounded"] is True
