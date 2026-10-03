"""Regression tests for app/guardrails.py.

Scoped deliberately to the deterministic safety-net logic — the part of
this project that's been hand-verified via live curl/browser testing over
and over across many sessions, with several real bugs only ever caught that
way (fuzzy-typo matching colliding with "help", a status report tripping
the email-claim check, etc.). Pinning those down here means a future change
that reintroduces one of them fails fast, instead of needing another full
manual regression pass through the running app.

Deliberately excludes anything that requires the actual LLM (agent.run_turn)
or a live Ollama/Chroma instance — those still need the manual/live checks
described in DEMO_QA_PREP.md.
"""

import datetime

import pytest

from app import guardrails


def at(hour: int) -> datetime.datetime:
    return datetime.datetime(2026, 1, 1, hour, 0)


class TestDeterministicPleasantryReply:
    def test_plain_greetings(self):
        for text in ["hi", "hello", "hey", "yo", "hiya", "hey buddy"]:
            assert guardrails.deterministic_pleasantry_reply(text) == guardrails.GREETING_REPLY

    def test_known_typos(self):
        # Specific typos actually seen live — see agent.py commit history.
        for text in ["hio", "good morinig", "mornign bro"]:
            assert guardrails.deterministic_pleasantry_reply(text) is not None

    def test_byes(self):
        for text in ["thanks", "thank you", "bye", "cya", "cheers"]:
            assert guardrails.deterministic_pleasantry_reply(text) == guardrails.BYE_REPLY

    def test_idiom_phrases(self):
        assert guardrails.deterministic_pleasantry_reply("how are you") == guardrails.GREETING_REPLY
        assert guardrails.deterministic_pleasantry_reply("hows it going bro") == guardrails.GREETING_REPLY

    def test_real_symptom_content_is_not_a_pleasantry(self):
        assert guardrails.deterministic_pleasantry_reply("i have a fever") is None
        assert guardrails.deterministic_pleasantry_reply("hi, i have a fever") is None

    def test_long_message_is_not_a_pleasantry_even_with_a_greeting_word(self):
        text = "hi there, i was wondering if you could help me understand this"
        assert guardrails.deterministic_pleasantry_reply(text) is None

    def test_false_positive_regressions(self):
        # Each of these previously broke under an earlier (reverted) fuzzy-
        # matching implementation of the typo tolerance above — "help" is
        # close enough to "helo" that a naive edit-distance check treated
        # "can you help" as a greeting. Pinned here so that mistake can't
        # silently come back.
        for text in [
            "him", "you", "can you help", "she said the fever is bad",
            "are you ok", "she is not doing well", "has it started",
            "the weather is nice", "was it good",
        ]:
            assert guardrails.deterministic_pleasantry_reply(text) is None

    def test_time_of_day_correction(self):
        # "good morning" said in the afternoon gets corrected, not echoed.
        reply = guardrails.deterministic_pleasantry_reply("good morning", now=at(14))
        assert reply.startswith("Good afternoon!")

        reply = guardrails.deterministic_pleasantry_reply("good morning", now=at(8))
        assert reply.startswith("Good morning!")

        reply = guardrails.deterministic_pleasantry_reply("good evening", now=at(8))
        assert reply.startswith("Good morning!")

    def test_time_of_day_falls_back_to_hi_at_night(self):
        reply = guardrails.deterministic_pleasantry_reply("good morning", now=at(2))
        assert reply.startswith("Hi!")

    def test_no_time_word_is_unaffected_by_clock(self):
        for hour in (2, 8, 14, 19):
            reply = guardrails.deterministic_pleasantry_reply("hi", now=at(hour))
            assert reply == guardrails.GREETING_REPLY

    def test_evening_bucket(self):
        reply = guardrails.deterministic_pleasantry_reply("good evening", now=at(18))
        assert reply.startswith("Good evening!")

    def test_message_with_no_letters_returns_none(self):
        assert guardrails.deterministic_pleasantry_reply("123") is None
        assert guardrails.deterministic_pleasantry_reply("!!!") is None


class TestCompletionClaims:
    def test_claims_order_placed(self):
        assert guardrails.claims_order_placed("Your order has been placed and is on its way.")
        assert guardrails.claims_order_placed("I'll go ahead and place the order for you now.")
        assert not guardrails.claims_order_placed("Would you like to order this?")

    def test_claims_email_sent(self):
        assert guardrails.claims_email_sent("A confirmation email has been sent to you.")
        assert not guardrails.claims_email_sent("Here are a couple of options for you.")

    def test_claims_address_saved(self):
        assert guardrails.claims_address_saved("I've saved your shipping address.")
        assert not guardrails.claims_address_saved("What's your shipping address?")


class TestCheckUnverifiedCompletion:
    def test_blocks_an_unbacked_order_claim(self):
        reply = "Your order has been placed and is being processed."
        result = guardrails.check_unverified_completion(reply, False, False, False)
        assert result == guardrails.FAKE_COMPLETION_GUARD_REPLY

    def test_allows_a_backed_order_claim(self):
        reply = "Your order has been placed and is being processed."
        result = guardrails.check_unverified_completion(reply, True, False, False)
        assert result is None

    def test_each_claim_is_tracked_independently(self):
        # A real order_placed:true doesn't make an accompanying, separate
        # "email has been sent" claim true too (see the function's own
        # docstring for the observed failure this covers).
        reply = "Your order has been placed. A confirmation email has been sent."
        result = guardrails.check_unverified_completion(reply, True, False, False)
        assert result == guardrails.FAKE_COMPLETION_GUARD_REPLY

    def test_allows_a_plain_reply_with_no_claims(self):
        result = guardrails.check_unverified_completion("Here are some options for you.", False, False, False)
        assert result is None


class TestHonestRepliesAreNotBlocked:
    """Real bug: the three claims_* checks used plain substring matching, so
    honest, truthful replies — "You haven't placed any orders yet", "I can't
    place the order until you give me an address", "Do you send a
    confirmation email?", even a health tip containing "in order to" — were
    replaced with the canned "what symptom is this for?" reply. The empty
    order history report is the one that matters most: check_order_status
    with no orders grounds nothing, so the model's correct answer was thrown
    away. All run with every real_* flag False (a no-tool turn)."""

    @staticmethod
    def _blocked(reply: str) -> bool:
        return guardrails.check_unverified_completion(reply, False, False, False) is not None

    def test_in_order_to_is_not_an_order_mention(self):
        assert not self._blocked("In order to reduce fever, take paracetamol. Doses should be placed at least 4 hours apart.")
        assert not self._blocked("Wait in order to get an accurate reading; once the beep is confirmed, read it.")

    def test_a_negated_order_report_is_not_a_claim(self):
        for reply in [
            "You haven't placed any orders yet.",
            "No orders have been placed yet. Would you like to order something?",
            "I can't place the order until you give me an address.",
            "Your order is not yet placed; what is your address?",
        ]:
            assert not self._blocked(reply), reply

    def test_a_negated_address_or_email_report_is_not_a_claim(self):
        assert not self._blocked("I haven't saved your address yet.")
        assert not self._blocked("No email sent yet - I don't have your email address.")

    def test_a_question_about_email_is_not_a_claim(self):
        assert not self._blocked("Do you send a confirmation email when I order?")

    def test_genuine_fabrications_are_still_blocked(self):
        # The safety net itself must not weaken: every one of these is a real
        # (or realistic) unbacked completion claim, including ones where a
        # negation-looking word sits AFTER the claim or in another sentence.
        for reply in [
            "Order confirmed! Order ID: ord-0002. Shipping to 123 First Rd",
            "I'll go ahead and place the order. Here's your order summary, once it's processed.",
            "Your order has been placed, no worries!",
            "Your order is confirmed — would you like another?",
            "I haven't had any trouble. Your order has been placed.",
            "No problem! Your order has been placed.",
            "I've saved your address. A confirmation email has been sent.",
            "Order placed for Cough Suppressant Syrup. Would you like to reorder this product in the future?",
        ]:
            assert self._blocked(reply), reply

    def test_a_negated_mention_followed_by_a_genuine_claim_is_still_blocked(self):
        # The first "placed" is negated, the second is a real claim — the
        # scan must keep going past a negated occurrence, not stop at it.
        assert self._blocked("I haven't placed any orders before. Your order has been placed.")


class TestOrderConfirmationTemplates:
    def test_build_order_confirmation_includes_key_fields(self):
        order = {
            "order_id": "ord-0001",
            "product_name": "Paracetamol 500mg Tablets",
            "quantity": 2,
            "total_price_usd": 9.98,
            "address": "123 Main St, Springfield",
            "email_sent": True,
        }
        text = guardrails.build_order_confirmation(order)
        assert "ord-0001" in text
        assert "Paracetamol 500mg Tablets" in text
        assert "123 Main St, Springfield" in text
        assert "9.98" in text
        assert "confirmation email has been sent" in text.lower()

    def test_build_order_confirmation_without_email_on_file(self):
        order = {
            "order_id": "ord-0002",
            "product_name": "Ibuprofen 200mg Tablets",
            "quantity": 1,
            "total_price_usd": 5.29,
            "address": "1 Test Way",
            "email_sent": False,
        }
        text = guardrails.build_order_confirmation(order)
        assert "reply with your email" in text.lower()

    def test_build_cancellation_confirmation(self):
        order = {"order_id": "ord-0003", "product_name": "Cetirizine 10mg Antihistamine Tablets"}
        text = guardrails.build_cancellation_confirmation(order)
        assert "ord-0003" in text
        assert "Cetirizine 10mg Antihistamine Tablets" in text
        assert "cancelled" in text.lower()


class TestLeakedToolIntent:
    TOOL_NAMES = {
        "lookup_symptom", "get_saved_address", "save_address", "start_order",
        "check_order_status", "cancel_order", "lookup_medicine_info", "decline_out_of_scope",
    }

    def test_detects_a_leaked_tool_name(self):
        text = "I'll then call the cancel_order tool with the correct order ID."
        assert guardrails.leaked_tool_intent(text, self.TOOL_NAMES) == "cancel_order"

    def test_plain_reply_has_no_leak(self):
        text = "Sure! What's your shipping address so I can send that out?"
        assert guardrails.leaked_tool_intent(text, self.TOOL_NAMES) is None


class TestReplyForDeferredOrder:
    def test_needs_address_confirmation_quotes_the_address_on_file(self):
        order = {"needs_address_confirmation": True, "address_on_file": "123 Main St, Springfield"}
        reply = guardrails.reply_for_deferred_order(order)
        assert "123 Main St, Springfield" in reply
        assert "ship to this address" in reply

    def test_no_address_on_file_asks_for_one(self):
        order = {"needs_address_confirmation": False}
        reply = guardrails.reply_for_deferred_order(order)
        assert "shipping address" in reply.lower()
        assert "123 Main St" not in reply

    def test_clamp_note_is_appended_when_quantity_was_capped(self):
        order = {"needs_address_confirmation": False, "quantity_clamped": "capped at our per-order limit of 2"}
        reply = guardrails.reply_for_deferred_order(order)
        assert "capped at our per-order limit of 2" in reply

    def test_no_clamp_note_when_quantity_was_not_capped(self):
        order = {"needs_address_confirmation": False, "quantity_clamped": None}
        reply = guardrails.reply_for_deferred_order(order)
        assert "capped" not in reply.lower()

    def test_clamp_note_uses_the_real_reason_for_a_non_positive_quantity(self):
        # Real bug fix: this used to always say "I've capped this at our
        # per-order limit of 2" whenever quantity_clamped was truthy, which
        # was flatly wrong for a non-positive quantity being raised to 1 —
        # that isn't the per-order limit at all.
        order = {"needs_address_confirmation": False, "quantity_clamped": "the requested quantity wasn't valid, so I used 1 instead"}
        reply = guardrails.reply_for_deferred_order(order)
        assert "wasn't valid" in reply
        assert "per-order limit" not in reply


class TestRecoverLeakedLookup:
    def test_recovers_a_real_matched_symptom(self):
        # The leaked JSON still names the real symptom the model meant to
        # look up — recovering it (rather than a dead-end "please rephrase")
        # is the whole point of this function.
        leaked = 'I\'ll call lookup_symptom with {"symptom": "fever"} now.'
        result = guardrails.recover_leaked_lookup(leaked)
        assert result is not None
        reply_text, result_json = result
        assert "Would you like to order one of these?" in reply_text
        assert "fev-001" in result_json or "fev-" in reply_text

    def test_unmatched_symptom_returns_the_out_of_scope_reply(self):
        leaked = '{"symptom": "a broken leg"}'
        reply_text, _ = guardrails.recover_leaked_lookup(leaked)
        assert reply_text == guardrails.OUT_OF_SCOPE_REPLY

    def test_no_symptom_pattern_returns_none(self):
        assert guardrails.recover_leaked_lookup("I'll call start_order now.") is None


@pytest.mark.usefixtures("isolated_db")
class TestRememberRecommendedProduct:
    def test_remembers_a_product_named_by_id(self):
        from app import store

        guardrails.remember_recommended_product("s1", "Paracetamol 500mg Tablets (fev-001) would be a good fit.")
        assert store.get_last_recommended_product("s1") == "fev-001"

    def test_remembers_a_product_named_only_by_full_catalog_name(self):
        from app import store

        # Real bug fix: a reply naming the product only by its human-
        # readable name (no id) used to leave this blind, and a later "yes"
        # fell through to the model's own memory, which ordered a
        # *different* product than the one actually shown.
        guardrails.remember_recommended_product("s1", "Paracetamol Extra Strength 650mg would be a good fit.")
        assert store.get_last_recommended_product("s1") == "fev-002"

    def test_multiple_products_mentioned_is_too_ambiguous_to_remember(self):
        from app import store

        guardrails.remember_recommended_product("s1", "Options: fev-001 or fev-002, either would work.")
        assert store.get_last_recommended_product("s1") is None

    def test_no_product_mentioned_remembers_nothing(self):
        from app import store

        guardrails.remember_recommended_product("s1", "Could you tell me more about your symptoms?")
        assert store.get_last_recommended_product("s1") is None
