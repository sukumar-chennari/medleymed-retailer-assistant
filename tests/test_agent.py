"""Regression tests for the deterministic clarifying-question state machine
in app/agent.py — arguably the most bug-prone logic in this project. Three
separate real bugs were found and fixed here through live testing before
any of this had unit-level coverage:

1. "dosage for paracetamol 500mg" (a golden eval query) triggered the fever
   child/adult clarifying question instead of ever reaching
   lookup_medicine_info, since "paracetamol" is a fever keyword.
2. "i think i have running nose and high temperature" went straight to an
   unfiltered product list (including the pediatric syrup) because the
   literal string "fever" never appeared in the message, even though it
   plainly classifies as fever.
3. "my child has a wet cough" (and the split-turn equivalent: "my child has
   a cough" -> asked dry/wet -> answered "wet") recommended the ADULT
   expectorant to a child, since cough's own dry/wet resolution had no way
   to know age had been mentioned.

Uses the isolated_db fixture (see conftest.py) for anything touching
resolve_clarification/_render_products, since those write session state
through store.py.
"""

import pytest

from app import agent


class TestIntentRegexes:
    def test_info_question_re(self):
        for text in [
            "dosage for paracetamol 500mg", "side effects of cetirizine",
            "what's in cold and flu relief", "how much can I take",
            "is it safe to take with food", "any warnings for this",
        ]:
            assert agent.INFO_QUESTION_RE.search(text), text

    def test_info_question_re_does_not_match_plain_symptom_text(self):
        assert not agent.INFO_QUESTION_RE.search("I have a fever and a headache")

    def test_order_intent_re(self):
        for text in ["order fev-001", "I'd like to buy this", "get me one of those"]:
            assert agent.ORDER_INTENT_RE.search(text), text

    def test_cancel_intent_re(self):
        assert agent.CANCEL_INTENT_RE.search("please cancel my order")
        assert not agent.CANCEL_INTENT_RE.search("I don't want this order anymore")

    def test_reorder_intent_re(self):
        for text in ["reorder that", "can you order that again", "order the same thing again"]:
            assert agent.REORDER_INTENT_RE.search(text), text
        assert not agent.REORDER_INTENT_RE.search("order fev-001")


class TestDetectAge:
    def test_detects_child(self):
        assert agent._detect_age("my child has a cough") == "child"
        assert agent._detect_age("it's for my toddler") == "child"

    def test_detects_adult(self):
        assert agent._detect_age("it's for myself") == "adult"

    def test_no_age_mentioned(self):
        assert agent._detect_age("i have a cough") == ""


class TestSplitTrigger:
    def test_splits_an_age_suffixed_trigger(self):
        assert agent._split_trigger("cough:child") == ("cough", "child")

    def test_plain_trigger_has_no_age(self):
        assert agent._split_trigger("fever") == ("fever", "")


class TestDetectCoughType:
    def test_detects_dry(self):
        assert agent._detect_cough_type("i have a dry cough") == "dry"

    def test_detects_wet_via_any_synonym(self):
        assert agent._detect_cough_type("bringing up mucus") == "wet"
        assert agent._detect_cough_type("a chesty cough") == "wet"

    def test_no_type_mentioned(self):
        assert agent._detect_cough_type("i have a cough") == ""


class TestNeedsClarification:
    def test_ambiguous_cough_needs_clarification(self):
        result = agent._needs_clarification("I have a cough")
        assert result is not None
        trigger, _ = result
        assert trigger == "cough"

    def test_cough_type_known_but_age_unknown_still_asks_encoding_the_type(self):
        # Real bug: this used to return the plain "cold" trigger, losing the
        # already-given "dry" qualifier — answering "adult" to that question
        # showed all 5 generic cold products instead of specifically the
        # dry-cough one. The type must survive in the trigger itself.
        trigger, question = agent._needs_clarification("I have a dry cough")
        assert trigger == "cold:cough_dry"
        assert "child" in question.lower()  # still the cold age question's wording

    def test_cough_wet_type_known_but_age_unknown(self):
        trigger, _ = agent._needs_clarification("bringing up mucus with my cough")
        assert trigger == "cold:cough_wet"

    def test_cough_with_age_encodes_it_in_the_trigger(self):
        trigger, _ = agent._needs_clarification("my child has a cough")
        assert trigger == "cough:child"

    def test_ambiguous_fever_needs_clarification(self):
        trigger, _ = agent._needs_clarification("I have a fever")
        assert trigger == "fever"

    def test_fever_with_qualifier_already_present_does_not_need_clarification(self):
        assert agent._needs_clarification("my child has a fever") is None

    def test_non_literal_category_match_still_triggers_clarification(self):
        # Real bug: no literal "fever" in this message, but it classifies as
        # fever via tools.classify_categories, and used to skip straight to
        # an unfiltered product list.
        result = agent._needs_clarification("i think i have running nose and high temperature")
        assert result is not None

    def test_info_question_never_needs_clarification(self):
        # Real bug: this exact golden-eval query used to trigger the fever
        # child/adult question instead of routing to lookup_medicine_info.
        assert agent._needs_clarification("dosage for paracetamol 500mg") is None
        assert agent._needs_clarification("side effects of cetirizine") is None

    def test_unrelated_message_does_not_need_clarification(self):
        assert agent._needs_clarification("what's the weather today") is None


@pytest.mark.usefixtures("isolated_db")
class TestResolveClarification:
    def test_cough_no_age_dry_resolves_to_adult_suppressant(self):
        reply = agent.resolve_clarification("cough", "dry", "s1")
        assert "Would you like to order this?" in reply
        # col-004 is the adult dry-cough suppressant.
        assert "Cough Suppressant" in reply

    def test_cough_no_age_wet_resolves_to_adult_expectorant(self):
        reply = agent.resolve_clarification("cough", "it's wet, bringing up mucus", "s1")
        assert "Guaifenesin" in reply

    def test_cough_child_dry_resolves_to_pediatric_product(self):
        # Real bug fix: this used to be indistinguishable from the adult
        # case and recommended the adult product to a child.
        reply = agent.resolve_clarification("cough:child", "dry", "s1")
        assert "Children's Cold & Cough Syrup" in reply

    def test_cough_child_wet_has_no_product_and_declines_safely(self):
        # Real bug fix: there is no pediatric wet-cough product in the
        # catalog — this must recommend seeing a pharmacist, never the
        # adult expectorant.
        reply = agent.resolve_clarification("cough:child", "wet", "s1")
        assert "pharmacist" in reply.lower()
        assert "Guaifenesin" not in reply

    def test_known_dry_cough_type_answered_adult_narrows_to_the_dry_product(self):
        # Real bug fix: this used to answer with the full 5-product generic
        # cold list, discarding the "dry" qualifier already given.
        reply = agent.resolve_clarification("cold:cough_dry", "adult", "s1")
        assert "Cough Suppressant" in reply
        assert "Cetirizine" not in reply  # not the generic 5-product list

    def test_known_wet_cough_type_answered_adult_narrows_to_the_wet_product(self):
        reply = agent.resolve_clarification("cold:cough_wet", "adult", "s1")
        assert "Guaifenesin" in reply

    def test_known_dry_cough_type_answered_child_resolves_to_pediatric_product(self):
        reply = agent.resolve_clarification("cold:cough_dry", "child", "s1")
        assert "Children's Cold & Cough Syrup" in reply

    def test_known_wet_cough_type_answered_child_declines_safely(self):
        reply = agent.resolve_clarification("cold:cough_wet", "child", "s1")
        assert "pharmacist" in reply.lower()

    def test_fever_child_resolves_to_pediatric_syrup(self):
        reply = agent.resolve_clarification("fever", "it's for my toddler", "s1")
        assert "Children's Paracetamol Syrup" in reply

    def test_unmatched_answer_returns_none(self):
        assert agent.resolve_clarification("cough", "maybe, not sure", "s1") is None

    def test_unknown_trigger_returns_none(self):
        assert agent.resolve_clarification("not-a-real-trigger", "dry", "s1") is None


@pytest.mark.usefixtures("isolated_db")
class TestResolvePrequalifiedClarification:
    def test_child_and_cough_type_in_the_same_message(self):
        # The exact real regression: age and cough-type both supplied in one
        # message, resolved directly without ever asking the dry/wet question.
        reply = agent._resolve_prequalified_clarification("my child has a wet cough", "s1")
        assert reply is not None
        assert "pharmacist" in reply.lower()

    def test_child_and_dry_cough_in_the_same_message(self):
        reply = agent._resolve_prequalified_clarification("my child has a dry cough", "s1")
        assert reply is not None
        assert "Children's Cold & Cough Syrup" in reply

    def test_info_question_is_never_prequalified(self):
        assert agent._resolve_prequalified_clarification("dosage for paracetamol 500mg", "s1") is None

    def test_message_with_no_qualifier_returns_none(self):
        assert agent._resolve_prequalified_clarification("I have a cough", "s1") is None
