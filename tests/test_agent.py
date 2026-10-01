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

    def test_has_cancel_intent_rejects_a_negated_cancel(self):
        # Real bug: CANCEL_INTENT_RE alone can't tell "cancel my order"
        # from "don't cancel my order" — both literally contain "cancel".
        assert agent._has_cancel_intent("please cancel my order")
        assert agent._has_cancel_intent("cancel my order please")
        for text in ["wait, don't cancel it", "no, do not cancel", "actually please dont cancel this"]:
            assert not agent._has_cancel_intent(text), text

    def test_reorder_intent_re(self):
        for text in ["reorder that", "can you order that again", "order the same thing again"]:
            assert agent.REORDER_INTENT_RE.search(text), text
        assert not agent.REORDER_INTENT_RE.search("order fev-001")

    def test_has_reorder_intent_rejects_a_negated_reorder(self):
        # Real bug: same negation-blindness class already fixed for
        # cancel intent — REORDER_INTENT_RE alone can't tell "reorder
        # that" from "don't reorder that".
        assert agent._has_reorder_intent("reorder that")
        assert agent._has_reorder_intent("please reorder the same thing again")
        for text in ["don't reorder that", "no, do not order that again"]:
            assert not agent._has_reorder_intent(text), text


class TestDetectAge:
    def test_detects_child(self):
        assert agent._detect_age("my child has a cough") == "child"
        assert agent._detect_age("it's for my toddler") == "child"

    def test_detects_adult(self):
        assert agent._detect_age("it's for myself") == "adult"

    def test_no_age_mentioned(self):
        assert agent._detect_age("i have a cough") == ""

    def test_substring_of_an_unrelated_word_does_not_count_as_an_answer(self):
        # Real bug: plain substring matching let "me" (a valid one-word
        # adult answer) match INSIDE unrelated words like "recommend" or
        # "medicine" that merely contain it — falsely detecting an age that
        # was never actually mentioned.
        assert agent._detect_age("what medicine do you recommend for a cough") == ""
        assert agent._detect_age("i have a cough, what do you recommend") == ""


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

    def test_a_deflection_word_does_not_falsely_encode_an_age(self):
        # Real bug: "what do you recommend" never mentions an age, but
        # "recommend" contains "me" as a substring — this used to encode
        # "cough:adult" as if age had genuinely been given.
        trigger, _ = agent._needs_clarification("I have a cough, what do you recommend")
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

    def test_message_describing_both_fever_and_cold_asks_one_combined_question(self):
        # Real bug: fever is checked before cold (dict order) and this used
        # to return on the first match alone, silently dropping cold — a
        # later age answer only ever recommended fever products.
        trigger, question = agent._needs_clarification("i think i have running nose and high temperature")
        assert trigger == "fever+cold"
        assert "child" in question.lower()

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

    def test_a_deflection_containing_me_as_a_substring_does_not_count_as_adult(self):
        # Real bug: "what do you recommend" never answers "child or adult?"
        # at all, but "recommend" contains "me" as a substring — plain
        # substring matching used to silently read this as answering
        # "adult" and immediately show adult products instead of falling
        # through to a normal turn.
        assert agent.resolve_clarification("fever", "what do you recommend", "s1") is None

    def test_me_as_a_real_standalone_word_still_counts_as_adult(self):
        reply = agent.resolve_clarification("fever", "me", "s1")
        assert "Paracetamol 500mg" in reply

    def test_unknown_trigger_returns_none(self):
        assert agent.resolve_clarification("not-a-real-trigger", "dry", "s1") is None

    def test_fever_and_cold_combined_trigger_merges_both_categories_adult(self):
        # Real bug fix: this used to only ever resolve fever (or cold,
        # depending on iteration order), never both.
        reply = agent.resolve_clarification("fever+cold", "for myself", "s1")
        assert "Paracetamol 500mg" in reply  # fever adult product
        assert "Cetirizine" in reply  # cold adult product

    def test_fever_and_cold_combined_trigger_merges_both_categories_child(self):
        reply = agent.resolve_clarification("fever+cold", "it's for my toddler", "s1")
        assert "Children's Paracetamol Syrup" in reply
        assert "Children's Cold & Cough Syrup" in reply

    def test_fever_and_cold_combined_trigger_unmatched_answer_returns_none(self):
        assert agent.resolve_clarification("fever+cold", "maybe, not sure", "s1") is None


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

    def test_fever_with_age_qualifier_in_the_same_message_resolves_directly(self):
        # No "cough" involved — exercises the generic per-trigger loop at
        # the end of the function, not the cough+age special case above it.
        reply = agent._resolve_prequalified_clarification("my baby has a fever", "s1")
        assert reply is not None
        assert "Children's Paracetamol Syrup" in reply

    def test_fever_and_cold_both_with_age_in_the_same_message_merges_both(self):
        # Real bug: age was given for both categories in one message, but
        # this used to resolve only whichever category matched first.
        reply = agent._resolve_prequalified_clarification("my baby has a runny nose and a fever", "s1")
        assert reply is not None
        assert "Children's Paracetamol Syrup" in reply
        assert "Children's Cold & Cough Syrup" in reply


@pytest.mark.usefixtures("isolated_db")
class TestKnownLimitationCoughFeverColdTripleCollision:
    """Pins the CURRENT, deliberately-unfixed behavior for a message that
    matches all three categories at once — see the "KNOWN, ACCEPTED
    LIMITATION" comment above FEVER_AND_COLD_COMBINED_QUESTION in agent.py
    for the full reasoning on why this is scoped out for now rather than
    rushed. These tests exist so a future change to this behavior is a
    deliberate decision, not an accidental regression nobody notices —
    none of the three sub-cases below lose safety information (no
    wrong-age/wrong-type product is ever recommended), only completeness."""

    def test_no_qualifiers_yet_asks_only_about_cough_type(self):
        trigger, question = agent._needs_clarification("I have a fever, cough, and runny nose")
        assert trigger == "cough"
        assert "dry" in question.lower()

    def test_no_qualifiers_yet_then_answering_resolves_only_cough(self):
        reply = agent.resolve_clarification("cough", "dry", "s1")
        assert "Cough Suppressant" in reply
        assert "Paracetamol" not in reply  # fever product silently dropped

    def test_age_given_without_cough_type_resolves_fever_and_cold_but_drops_cough(self):
        reply = agent._resolve_prequalified_clarification("my child has a fever, cough, and runny nose", "s1")
        assert reply is not None
        assert "Children's Paracetamol Syrup" in reply
        assert "Children's Cold & Cough Syrup" in reply

    def test_age_and_cough_type_both_given_resolves_only_cough(self):
        reply = agent._resolve_prequalified_clarification("my child has a dry cough, a fever, and runny nose", "s1")
        assert reply is not None
        assert "Children's Cold & Cough Syrup" in reply
        assert "Children's Paracetamol Syrup" not in reply  # fever product silently dropped
