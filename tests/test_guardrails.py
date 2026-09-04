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
