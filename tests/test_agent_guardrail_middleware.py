"""Tests for _GuardrailMiddleware's two hooks — wrap_tool_call (pre-tool-
execution: blocks, redirects, or corrects a tool call before it runs) and
after_agent (post-execution: overrides the model's final reply text).
Together these are arguably the single most safety-critical piece of
logic in the whole project (every "blocked_X"/"declined_forced"/
"leaked_tool_intent"/citation-append guard mentioned throughout agent.py's
other docstrings lives in one of these two methods), yet neither had ever
been unit tested — only ever exercised live via curl, conversation_eval.py,
or manual browser testing.

Both are cheaply testable without any live LLM: _GuardrailMiddleware.
__init__ takes plain parameters with no LangChain agent dependency at all;
wrap_tool_call only needs request.tool_call (a dict) and a callable
handler (both faked below); after_agent's `runtime` parameter is entirely
unused in its body, and `state` only needs state["messages"][-1] to have
.content/.id, which a real AIMessage trivially provides.

Uses the isolated_db fixture since several branches read/write session
state and metrics through store.py.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app import agent, guardrails, store

pytestmark = pytest.mark.usefixtures("isolated_db")


def _middleware(
    session_id="s1",
    user_text="",
    image_read=False,
    had_shown_recommendation=False,
    symptom_lookup_grounded=False,
    turn_state=None,
):
    return agent._GuardrailMiddleware(
        session_id=session_id,
        user_text=user_text,
        image_read=image_read,
        had_shown_recommendation=had_shown_recommendation,
        symptom_lookup_grounded=symptom_lookup_grounded,
        turn_state=turn_state if turn_state is not None else {},
    )


class _FakeRequest:
    def __init__(self, name, args, call_id="call-1"):
        self.tool_call = {"name": name, "args": args, "id": call_id}


def _handler_returning(content):
    calls = []

    def handler(request):
        calls.append(request)
        return SimpleNamespace(content=content)

    handler.calls = calls
    return handler


def _raising_handler(exc):
    def handler(request):
        raise exc

    return handler


class TestDeclineOutOfScope:
    def test_grounded_by_current_text_redirects_to_a_real_lookup_instead_of_the_handler(self):
        mw = _middleware(user_text="I have a fever")
        handler = _handler_returning("unused")
        result = mw.wrap_tool_call(_FakeRequest("decline_out_of_scope", {}), handler)
        assert not handler.calls
        parsed = json.loads(result.content)
        assert parsed["matched"] is True
        assert parsed["category"] == "fever"
        assert store.get_last_products("s1") is not None

    def test_grounded_by_an_established_category_plus_a_continuation_word(self):
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        mw = _middleware(session_id="s1", user_text="any alternative?")
        result = mw.wrap_tool_call(_FakeRequest("decline_out_of_scope", {}), _handler_returning("unused"))
        parsed = json.loads(result.content)
        assert parsed["matched"] is True
        assert parsed["category"] == "fever"

    def test_established_category_alone_without_a_continuation_word_stays_declined(self):
        # Real bug this guards: a bare "a category exists somewhere in this
        # session" was wrongly overriding a CORRECT decline for genuinely
        # unrelated text like "who is narendra modi".
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        turn_state = {}
        mw = _middleware(session_id="s1", user_text="who is narendra modi", turn_state=turn_state)
        result = mw.wrap_tool_call(_FakeRequest("decline_out_of_scope", {}), _handler_returning("unused"))
        assert turn_state["declined_forced"] is True
        assert "out of scope" in result.content

    def test_a_negated_continuation_word_stays_declined(self):
        # Real bug: same negation-blindness class already fixed for
        # cancel_order/reorder_last/start_order — "don't show me anything
        # else" literally contains "else", which used to override a
        # legitimate decline and force an unrelated product list back in.
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        turn_state = {}
        mw = _middleware(session_id="s1", user_text="don't show me anything else, I'm done", turn_state=turn_state)
        result = mw.wrap_tool_call(_FakeRequest("decline_out_of_scope", {}), _handler_returning("unused"))
        assert turn_state["declined_forced"] is True
        assert "out of scope" in result.content


class TestLookupSymptom:
    def test_info_question_redirects_to_medicine_info_instead(self):
        mw = _middleware(user_text="dosage for paracetamol 500mg", symptom_lookup_grounded=True)
        handler = _handler_returning("unused")
        result = mw.wrap_tool_call(_FakeRequest("lookup_symptom", {"symptom": "fever"}), handler)
        assert not handler.calls
        parsed = json.loads(result.content)
        assert parsed["results"]
        assert parsed["results"][0]["product"] == "Paracetamol 500mg Tablets"

    def test_an_unread_photo_does_not_stop_established_category_reuse(self):
        # With no usable photo and no symptom text, an earlier-established
        # category is the only grounding, so a model-invented category switch
        # is corrected back to it (handler never runs).
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        mw = _middleware(session_id="s1", user_text="what is this?", image_read=False, symptom_lookup_grounded=True)
        handler = _handler_returning("unused")
        result = mw.wrap_tool_call(_FakeRequest("lookup_symptom", {"symptom": "cold"}), handler)
        assert not handler.calls
        assert json.loads(result.content)["category"] == "fever"

    def test_a_read_photo_grounds_the_lookup_itself(self):
        # A photo that WAS read is real grounding for whatever the model asks
        # about, so the established-category correction is skipped and the
        # model's own lookup runs.
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        mw = _middleware(session_id="s1", user_text="what is this?", image_read=True, symptom_lookup_grounded=True)
        handler = _handler_returning(json.dumps({"matched": True, "category": "cold", "products": []}))
        mw.wrap_tool_call(_FakeRequest("lookup_symptom", {"symptom": "cold"}), handler)
        assert handler.calls

    def test_blocked_when_not_grounded(self):
        turn_state = {}
        mw = _middleware(user_text="fix my code", symptom_lookup_grounded=False, turn_state=turn_state)
        handler = _handler_returning("unused")
        result = mw.wrap_tool_call(_FakeRequest("lookup_symptom", {"symptom": "fever"}), handler)
        assert not handler.calls
        assert turn_state["blocked_ungrounded_lookup"] is True
        assert json.loads(result.content)["matched"] is False

    def test_established_category_overrides_a_model_switched_category(self):
        # Real bug: asked "last one" right after a fever product list, the
        # model called lookup_symptom(symptom="cold") with nothing in the
        # message suggesting cold at all.
        agent._remember_products("s1", '{"matched": true, "products": [{"id": "fev-001"}]}')
        mw = _middleware(session_id="s1", user_text="last one", symptom_lookup_grounded=True)
        handler = _handler_returning("unused")
        result = mw.wrap_tool_call(_FakeRequest("lookup_symptom", {"symptom": "cold"}), handler)
        assert not handler.calls
        assert json.loads(result.content)["category"] == "fever"

    def test_grounded_call_passes_through_to_the_real_handler(self):
        mw = _middleware(user_text="I have a fever", symptom_lookup_grounded=True)
        handler = _handler_returning(json.dumps({"matched": True, "products": [{"id": "fev-001"}]}))
        mw.wrap_tool_call(_FakeRequest("lookup_symptom", {"symptom": "fever"}), handler)
        assert handler.calls
        assert store.get_last_products("s1") == [{"id": "fev-001"}]


class TestStartOrder:
    def test_blocked_when_nothing_was_recommended_and_no_order_intent(self):
        mw = _middleware(user_text="ok", had_shown_recommendation=False)
        handler = _handler_returning("unused")
        result = mw.wrap_tool_call(_FakeRequest("start_order", {"product_id": "fev-001"}), handler)
        assert not handler.calls
        assert json.loads(result.content)["order_placed"] is False

    def test_allowed_when_a_recommendation_was_already_shown(self):
        turn_state = {}
        mw = _middleware(user_text="yes", had_shown_recommendation=True, turn_state=turn_state)
        handler = _handler_returning(json.dumps({"order_placed": True, "order_id": "ord-0001", "email_sent": True}))
        mw.wrap_tool_call(_FakeRequest("start_order", {"product_id": "fev-001"}), handler)
        assert turn_state["real_order_placed"] is True
        assert turn_state["completed_order"]["order_id"] == "ord-0001"
        assert turn_state["real_email_sent"] is True

    def test_allowed_via_explicit_order_intent_even_without_a_shown_recommendation(self):
        mw = _middleware(user_text="order fev-001", had_shown_recommendation=False)
        handler = _handler_returning(json.dumps({"order_placed": False, "needs_address_confirmation": True}))
        result = mw.wrap_tool_call(_FakeRequest("start_order", {"product_id": "fev-001"}), handler)
        assert handler.calls
        assert result.content

    def test_blocked_when_order_intent_is_negated(self):
        # Real bug: same negation-blindness class already fixed for
        # cancel_order/reorder_last — "no, don't order that for me, I just
        # want info" literally contains "order", which used to read as
        # confirmed order intent and let start_order through unblocked
        # even with no recommendation shown yet.
        mw = _middleware(user_text="no, don't order that for me, I just want info", had_shown_recommendation=False)
        handler = _handler_returning(json.dumps({"order_placed": True, "order_id": "ord-0001"}))
        result = mw.wrap_tool_call(_FakeRequest("start_order", {"product_id": "fev-001"}), handler)
        assert not handler.calls
        assert json.loads(result.content)["order_placed"] is False

    def test_deferred_order_is_recorded_in_turn_state(self):
        turn_state = {}
        mw = _middleware(user_text="yes", had_shown_recommendation=True, turn_state=turn_state)
        handler = _handler_returning(json.dumps({"order_placed": False, "needs_address_confirmation": True}))
        mw.wrap_tool_call(_FakeRequest("start_order", {"product_id": "fev-001"}), handler)
        assert turn_state["deferred_order"]["needs_address_confirmation"] is True

    def test_an_error_result_is_not_recorded_as_a_deferred_order(self):
        turn_state = {}
        mw = _middleware(user_text="yes", had_shown_recommendation=True, turn_state=turn_state)
        handler = _handler_returning(json.dumps({"error": "Unknown product_id"}))
        mw.wrap_tool_call(_FakeRequest("start_order", {"product_id": "bad-id"}), handler)
        assert "deferred_order" not in turn_state
        assert "completed_order" not in turn_state

    def test_non_json_handler_response_does_not_raise(self):
        # try/except (TypeError, ValueError) fallback — a malformed tool
        # result must not crash the whole turn. parsed_order falls back to
        # {}, which correctly reads as "deferred, not completed" (no
        # "error" key, no "order_placed" either) rather than raising.
        turn_state = {}
        mw = _middleware(user_text="yes", had_shown_recommendation=True, turn_state=turn_state)
        handler = _handler_returning("not valid json")
        mw.wrap_tool_call(_FakeRequest("start_order", {"product_id": "fev-001"}), handler)
        assert "completed_order" not in turn_state
        assert turn_state["deferred_order"] == {}


class TestReorderLast:
    def test_blocked_without_reorder_intent(self):
        mw = _middleware(user_text="ok")
        handler = _handler_returning("unused")
        result = mw.wrap_tool_call(_FakeRequest("reorder_last", {}), handler)
        assert not handler.calls
        assert json.loads(result.content)["order_placed"] is False

    def test_allowed_with_explicit_reorder_intent_records_the_result(self):
        turn_state = {}
        mw = _middleware(user_text="reorder that", turn_state=turn_state)
        handler = _handler_returning(json.dumps({"order_placed": True, "order_id": "ord-0002"}))
        mw.wrap_tool_call(_FakeRequest("reorder_last", {}), handler)
        assert turn_state["real_order_placed"] is True
        assert turn_state["completed_order"]["order_id"] == "ord-0002"

    def test_blocked_when_reorder_is_negated(self):
        # Real bug: same negation-blindness class already fixed for
        # cancel_order — "don't reorder that" literally contains "reorder",
        # which used to read as confirmed reorder intent.
        for text in ["don't reorder that", "no, do not order that again"]:
            handler = _handler_returning("unused")
            mw = _middleware(user_text=text)
            result = mw.wrap_tool_call(_FakeRequest("reorder_last", {}), handler)
            assert not handler.calls, text
            assert json.loads(result.content)["order_placed"] is False, text

    def test_email_sent_is_grounded_too(self):
        turn_state = {}
        mw = _middleware(user_text="reorder that", turn_state=turn_state)
        handler = _handler_returning(json.dumps({"order_placed": True, "order_id": "ord-0002", "email_sent": True}))
        mw.wrap_tool_call(_FakeRequest("reorder_last", {}), handler)
        assert turn_state["real_email_sent"] is True

    def test_non_json_handler_response_does_not_raise(self):
        turn_state = {}
        mw = _middleware(user_text="reorder that", turn_state=turn_state)
        handler = _handler_returning("not valid json")
        mw.wrap_tool_call(_FakeRequest("reorder_last", {}), handler)
        assert "completed_order" not in turn_state
        assert turn_state["deferred_order"] == {}


class TestCancelOrder:
    def test_blocked_without_cancel_intent(self):
        mw = _middleware(user_text="hmm not sure")
        handler = _handler_returning("unused")
        result = mw.wrap_tool_call(_FakeRequest("cancel_order", {}), handler)
        assert not handler.calls
        assert json.loads(result.content)["cancelled"] is False

    def test_allowed_with_cancel_intent_records_the_cancelled_order(self):
        turn_state = {}
        mw = _middleware(user_text="cancel my order", turn_state=turn_state)
        handler = _handler_returning(json.dumps({"cancelled": True, "order_id": "ord-0001"}))
        mw.wrap_tool_call(_FakeRequest("cancel_order", {}), handler)
        assert turn_state["cancelled_order"]["order_id"] == "ord-0001"

    def test_blocked_when_cancel_is_negated(self):
        # Real bug: CANCEL_INTENT_RE alone can't tell "cancel my order"
        # from "don't cancel my order" — both contain the word "cancel".
        # A user retracting a cancellation used to read as confirmed
        # intent, letting a real cancel_order call straight through the
        # one guardrail that exists specifically to never trust the
        # model's own judgment for this state-changing action.
        for text in ["wait, don't cancel it", "no, do not cancel", "actually please dont cancel this"]:
            handler = _handler_returning("unused")
            mw = _middleware(user_text=text)
            result = mw.wrap_tool_call(_FakeRequest("cancel_order", {}), handler)
            assert not handler.calls, text
            assert json.loads(result.content)["cancelled"] is False, text

    def test_non_json_handler_response_does_not_raise(self):
        turn_state = {}
        mw = _middleware(user_text="cancel my order", turn_state=turn_state)
        handler = _handler_returning("not valid json")
        mw.wrap_tool_call(_FakeRequest("cancel_order", {}), handler)
        assert "cancelled_order" not in turn_state


class TestSaveAddress:
    def test_saved_true_grounds_the_address_saved_claim(self):
        turn_state = {}
        mw = _middleware(turn_state=turn_state)
        handler = _handler_returning(json.dumps({"saved": True}))
        mw.wrap_tool_call(_FakeRequest("save_address", {"address": "123 Main St"}), handler)
        assert turn_state["real_address_saved"] is True

    def test_saved_false_does_not_ground_anything(self):
        turn_state = {}
        mw = _middleware(turn_state=turn_state)
        handler = _handler_returning(json.dumps({"saved": False}))
        mw.wrap_tool_call(_FakeRequest("save_address", {}), handler)
        assert "real_address_saved" not in turn_state

    def test_non_json_handler_response_does_not_raise(self):
        turn_state = {}
        mw = _middleware(turn_state=turn_state)
        handler = _handler_returning("not valid json")
        mw.wrap_tool_call(_FakeRequest("save_address", {}), handler)
        assert "real_address_saved" not in turn_state


class TestLookupMedicineInfo:
    def test_tracks_the_retrieval_score_and_sources_from_the_handlers_response(self):
        turn_state = {}
        mw = _middleware(session_id="s1", turn_state=turn_state)
        handler = _handler_returning(
            json.dumps({"results": [{"source": "fev-001.md", "section": "Dosage", "score": 0.9}]})
        )
        mw.wrap_tool_call(_FakeRequest("lookup_medicine_info", {"query": "dosage"}), handler)
        assert turn_state["retrieval_score"] == 0.9
        assert turn_state["retrieval_sources"] == [("fev-001.md", "Dosage")]

    def test_non_json_handler_response_does_not_raise(self):
        turn_state = {}
        mw = _middleware(session_id="s1", turn_state=turn_state)
        handler = _handler_returning("not valid json")
        mw.wrap_tool_call(_FakeRequest("lookup_medicine_info", {"query": "dosage"}), handler)
        assert "retrieval_score" not in turn_state
        assert "retrieval_sources" not in turn_state


class TestCheckOrderStatus:
    def test_has_orders_grounds_all_three_completion_flags_together(self):
        turn_state = {}
        mw = _middleware(turn_state=turn_state)
        handler = _handler_returning(json.dumps({"orders": [{"order_id": "ord-0001"}]}))
        mw.wrap_tool_call(_FakeRequest("check_order_status", {}), handler)
        assert turn_state["real_order_placed"] is True
        assert turn_state["real_email_sent"] is True
        assert turn_state["real_address_saved"] is True

    def test_no_orders_does_not_ground_anything(self):
        turn_state = {}
        mw = _middleware(turn_state=turn_state)
        handler = _handler_returning(json.dumps({"orders": []}))
        mw.wrap_tool_call(_FakeRequest("check_order_status", {}), handler)
        assert "real_order_placed" not in turn_state

    def test_non_json_handler_response_does_not_raise(self):
        turn_state = {}
        mw = _middleware(turn_state=turn_state)
        handler = _handler_returning("not valid json")
        mw.wrap_tool_call(_FakeRequest("check_order_status", {}), handler)
        assert "real_order_placed" not in turn_state


class TestGenericFallbackAndErrorHandling:
    def test_an_unrecognized_tool_name_passes_straight_through_to_the_handler(self):
        mw = _middleware()
        handler = _handler_returning(json.dumps({"address": None}))
        result = mw.wrap_tool_call(_FakeRequest("get_saved_address", {"user_id": "demo_user"}), handler)
        assert handler.calls
        assert result.content == json.dumps({"address": None})

    def test_a_handler_exception_is_caught_and_reported_not_raised(self):
        mw = _middleware()
        handler = _raising_handler(RuntimeError("boom"))
        result = mw.wrap_tool_call(_FakeRequest("get_saved_address", {}), handler)
        assert "Error calling get_saved_address" in result.content
        assert "boom" in result.content


def _after_agent(mw, reply_text: str) -> str:
    state = {"messages": [AIMessage(content=reply_text, id="msg-1")]}
    result = mw.after_agent(state, runtime=None)
    return result["messages"][0].content


class TestAfterAgentUnverifiedCancellation:
    def test_an_ungrounded_cancellation_claim_is_replaced_and_logged(self):
        mw = _middleware(turn_state={})
        assert _after_agent(mw, "I've cancelled your order ord-0001.") == guardrails.CANCELLATION_NOT_DONE_REPLY
        names = {e["name"] for e in store.get_metrics_summary()["recent_guardrail_events"]}
        assert "unverified_cancellation_blocked" in names

    def test_it_also_catches_a_claim_after_the_cancel_tool_call_was_blocked(self):
        # blocked_unconfirmed_cancel stops the TOOL; the model can still SAY it
        # was done. Nothing in turn_state records a real cancellation.
        mw = _middleware(user_text="hmm", turn_state={})
        assert _after_agent(mw, "Done, order cancelled!") == guardrails.CANCELLATION_NOT_DONE_REPLY

    def test_a_real_cancellation_still_renders_the_apps_own_confirmation(self):
        order = {"order_id": "ord-0001", "product_name": "Paracetamol 500mg Tablets"}
        mw = _middleware(turn_state={"cancelled_order": order})
        assert _after_agent(mw, "whatever the model says") == guardrails.build_cancellation_confirmation(order)

    def test_a_status_listing_that_mentions_cancelled_is_not_blocked(self):
        # check_order_status returning real orders sets real_order_placed.
        mw = _middleware(turn_state={"real_order_placed": True})
        reply = "ord-0001 has been cancelled; ord-0002 is on its way."
        assert _after_agent(mw, reply) == reply

    def test_an_offer_to_cancel_is_untouched(self):
        mw = _middleware(turn_state={})
        reply = "Would you like me to cancel your order?"
        assert _after_agent(mw, reply) == reply


class TestAfterAgentStructuralOverrides:
    def test_blocked_ungrounded_lookup_forces_the_out_of_scope_reply(self):
        mw = _middleware(turn_state={"blocked_ungrounded_lookup": True})
        assert _after_agent(mw, "here are some cold products anyway") == guardrails.OUT_OF_SCOPE_REPLY

    def test_declined_forced_forces_the_out_of_scope_reply(self):
        mw = _middleware(turn_state={"declined_forced": True})
        assert _after_agent(mw, "anything the model said") == guardrails.OUT_OF_SCOPE_REPLY

    def test_completed_order_renders_the_real_confirmation_not_the_models_text(self):
        order = {
            "order_id": "ord-0001", "product_name": "Paracetamol 500mg Tablets", "quantity": 1,
            "total_price_usd": 4.99, "address": "123 Main St", "email_sent": False,
        }
        mw = _middleware(turn_state={"completed_order": order})
        result = _after_agent(mw, "some vague model confirmation")
        assert "Order confirmed!" in result
        assert "ord-0001" in result

    def test_cancelled_order_renders_the_real_cancellation_not_the_models_text(self):
        order = {"order_id": "ord-0002", "product_name": "Cough Suppressant Syrup"}
        mw = _middleware(turn_state={"cancelled_order": order})
        result = _after_agent(mw, "some vague model text")
        assert "cancelled" in result.lower()
        assert "ord-0002" in result

    def test_leaked_decline_out_of_scope_is_forced_to_the_canned_reply(self):
        mw = _middleware()
        result = _after_agent(mw, "I'll call decline_out_of_scope to end this conversation")
        assert result == guardrails.OUT_OF_SCOPE_REPLY

    def test_leaked_lookup_symptom_is_recovered_into_a_real_product_list(self):
        mw = _middleware(session_id="s1")
        leaked = 'I\'ll call lookup_symptom with {"symptom": "fever"} now.'
        result = _after_agent(mw, leaked)
        assert "Would you like to order one of these?" in result
        assert store.get_last_products("s1") is not None

    def test_leaked_lookup_symptom_that_cannot_be_recovered_falls_back_to_the_guard_reply(self):
        mw = _middleware()
        result = _after_agent(mw, "I'll call lookup_symptom now.")
        assert result == guardrails.FAKE_COMPLETION_GUARD_REPLY

    def test_a_leaked_unrelated_tool_name_is_replaced_with_the_guard_reply(self):
        mw = _middleware()
        result = _after_agent(mw, "I'll call start_order with product_id 'fev-001'.")
        assert result == guardrails.FAKE_COMPLETION_GUARD_REPLY

    def test_deferred_order_is_rendered_from_the_real_data_not_the_models_text(self):
        mw = _middleware(turn_state={"deferred_order": {"needs_address_confirmation": False}})
        result = _after_agent(mw, "some vague model text about needing an address")
        assert "shipping address" in result.lower()

    def test_plain_ungrounded_reply_passes_through_untouched(self):
        mw = _middleware()
        assert _after_agent(mw, "Here are some options for you.") == "Here are some options for you."

    def test_an_unbacked_completion_claim_is_blocked(self):
        mw = _middleware(turn_state={"real_order_placed": False})
        result = _after_agent(mw, "Your order has been placed and is being processed.")
        assert result == guardrails.FAKE_COMPLETION_GUARD_REPLY

    def test_a_backed_completion_claim_is_allowed_through(self):
        mw = _middleware(turn_state={"real_order_placed": True})
        reply = "Your order has been placed and is being processed."
        assert _after_agent(mw, reply) == reply

    def test_missing_citation_is_appended_from_the_real_retrieved_sources(self):
        # Real bug found by conversation_eval.py: a correctly-grounded
        # answer sometimes describes its source in prose instead of the
        # "(Source: ...)" format the system prompt asks for.
        mw = _middleware(turn_state={"retrieval_sources": [("fev-001.md", "Dosage")]})
        result = _after_agent(mw, "Take 1-2 tablets every 4-6 hours.")
        assert "(Source: fev-001.md § Dosage)" in result

    def test_an_existing_citation_is_not_duplicated(self):
        mw = _middleware(turn_state={"retrieval_sources": [("fev-001.md", "Dosage")]})
        reply = "Take 1-2 tablets every 4-6 hours. (Source: fev-001.md § Dosage)"
        result = _after_agent(mw, reply)
        assert result.count("(Source:") == 1

    def test_retrieval_confidence_is_appended_when_present(self):
        mw = _middleware(turn_state={"retrieval_score": 0.86})
        result = _after_agent(mw, "Take 1-2 tablets every 4-6 hours.")
        assert "Retrieval confidence: 86%" in result

    def test_citation_and_confidence_can_both_be_appended_in_one_reply(self):
        mw = _middleware(turn_state={"retrieval_sources": [("fev-001.md", "Dosage")], "retrieval_score": 0.86})
        result = _after_agent(mw, "Take 1-2 tablets every 4-6 hours.")
        assert "(Source: fev-001.md § Dosage)" in result
        assert "Retrieval confidence: 86%" in result

    def test_a_guard_overridden_reply_never_gets_a_citation_or_confidence_bolted_on(self):
        # Real bug this guards: a confidence number bolted onto a guard-
        # overridden generic reply wouldn't mean anything, since that
        # reply isn't a RAG-grounded answer.
        mw = _middleware(turn_state={"real_order_placed": False, "retrieval_score": 0.9})
        result = _after_agent(mw, "Your order has been placed.")
        assert result == guardrails.FAKE_COMPLETION_GUARD_REPLY
        assert "Retrieval confidence" not in result


class TestInvokeAgentWithRetry:
    """_invoke_agent_with_retry takes agent_graph as a plain parameter — a
    fake with a scripted .invoke() is enough to test the retry logic
    without ever needing a real agent graph or LLM."""

    def test_succeeds_on_the_first_attempt(self):
        calls = []
        graph = SimpleNamespace(invoke=lambda payload, config: calls.append(1) or "result")
        result = agent._invoke_agent_with_retry(graph, {}, {})
        assert result == "result"
        assert len(calls) == 1

    def test_retries_once_after_a_transient_failure_then_succeeds(self):
        attempts = {"n": 0}

        def invoke(payload, config):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient Ollama failure")
            return "result"

        graph = SimpleNamespace(invoke=invoke)
        result = agent._invoke_agent_with_retry(graph, {}, {})
        assert result == "result"
        assert attempts["n"] == 2

    def test_raises_the_last_exception_if_both_attempts_fail(self):
        def invoke(payload, config):
            raise RuntimeError("still failing")

        graph = SimpleNamespace(invoke=invoke)
        with pytest.raises(RuntimeError, match="still failing"):
            agent._invoke_agent_with_retry(graph, {}, {})

    def test_a_recursion_limit_hit_is_never_retried(self):
        from langgraph.errors import GraphRecursionError

        attempts = {"n": 0}

        def invoke(payload, config):
            attempts["n"] += 1
            raise GraphRecursionError("recursion limit reached")

        graph = SimpleNamespace(invoke=invoke)
        with pytest.raises(GraphRecursionError):
            agent._invoke_agent_with_retry(graph, {}, {})
        assert attempts["n"] == 1
