"""Stated conditions are checked against what each product's own information
says (app/contraindications.py).

Real bugs (Oct 21 live probe): "I have a fever and stomach ulcer" and "I'm on
warfarin ... fever" were offered Ibuprofen 200mg Tablets (whose information says
it isn't recommended with a history of stomach ulcers); "high blood pressure and
a stuffy nose" was recommended Pseudoephedrine; a stated ibuprofen allergy did
not keep ibuprofen out of the lists."""

import json

import pytest
from fastapi.testclient import TestClient

from app import agent, config, contraindications as ci, main, store, tools

CATALOG = store.get_catalog()
FEVER_LIST = (
    "Here's what I'd recommend:\n\n"
    "- Paracetamol 500mg Tablets (fev-001) — Reduces fever.\n"
    "- Paracetamol Extra Strength 650mg (fev-002) — Higher-dose fever reducer.\n"
    "- Ibuprofen 200mg Tablets (fev-003) — Reduces fever and inflammation.\n\n"
    "Which one would you like to try?\n\nThis is general OTC guidance."
)


class TestConditions:
    @pytest.mark.parametrize("text, key", [
        ("I have a fever and stomach ulcer", "ulcer"), ("I have kidney disease", "kidney"),
        ("I have a heart condition", "heart"), ("I have high blood pressure", "high_bp"),
        ("I have a thyroid problem", "thyroid"), ("I'm pregnant", "pregnant"), ("I have asthma", "asthma"),
        ("I'm on warfarin", "blood_thinner"), ("I take aspirin daily", "nsaid_use"),
        ("I'm taking paracetamol already", "paracetamol_use"), ("I'm on an MAOI", "maoi"),
        ("I have liver problems", "liver"), ("I drink alcohol regularly", "alcohol"),
    ])
    def test_each_condition_is_recognised(self, text, key):
        assert key in ci.conditions_in(text)

    @pytest.mark.parametrize("text", [
        "no history of ulcers", "I have no kidney problems", "I don't have asthma", "I have a cold and a runny nose",
        "I have a fever", "my kidney bean salad was great",
    ])
    def test_negated_or_absent_conditions_are_ignored(self, text):
        assert ci.conditions_in(text) == set()

    def test_allergies_map_to_every_product_containing_the_ingredient(self):
        assert set(ci.allergic_product_ids("I'm allergic to ibuprofen")) == {"fev-003"}
        assert set(ci.allergic_product_ids("paracetamol allergy")) == {"fev-001", "fev-002", "fev-004", "col-003"}
        assert "fev-003" in ci.allergic_product_ids("I'm allergic to aspirin or paracetamol")
        assert ci.allergic_product_ids("I'm allergic to penicillin") == {}
        assert ci.allergic_product_ids("I'm not allergic to ibuprofen") == {}


class TestRules:
    def test_every_rule_names_a_real_product(self):
        ids = {p["id"] for p in CATALOG}
        assert {pid for _, pid, _, _ in ci._RULES} <= ids

    def test_ibuprofen_is_ruled_out_by_what_its_own_information_lists(self):
        for text in ["stomach ulcer", "kidney problems", "a heart condition", "I'm pregnant", "I'm on warfarin"]:
            assert any(m == "exclude" for m, _ in ci.concerns(f"I have {text}" if "I'm" not in text else text).get("fev-003", [])), text

    def test_pseudoephedrine_and_cold_flu_are_ruled_out_by_high_blood_pressure(self):
        got = ci.concerns("I have high blood pressure")
        assert "col-002" in got and "col-003" in got and "fev-003" not in got


class TestScreenReply:
    def test_the_ruled_out_bullet_is_removed_and_explained(self):
        reply, excluded = ci.screen_reply(FEVER_LIST, "I have a fever and stomach ulcer", CATALOG)
        assert "(fev-003)" not in reply.split("Left out")[0]
        assert "(fev-001)" in reply and "(fev-002)" in reply
        assert "Left out because of what you told me" in reply and "stomach ulcer" in reply
        assert excluded == {"fev-003"}

    def test_a_note_is_added_when_the_information_only_says_to_be_careful(self):
        reply, excluded = ci.screen_reply(FEVER_LIST, "I have asthma", CATALOG)
        assert "(fev-003)" in reply and excluded == set()
        assert "Note on Ibuprofen 200mg Tablets" in reply

    def test_a_single_ruled_out_recommendation_becomes_a_refusal(self):
        single = "Ibuprofen 200mg Tablets would be a good fit — Reduces fever.\n\nWould you like to order this?"
        reply, excluded = ci.screen_reply(single, "I have a stomach ulcer", CATALOG)
        assert reply.startswith("I can't recommend Ibuprofen 200mg Tablets for you")
        assert excluded == {"fev-003"}

    def test_a_list_where_everything_is_ruled_out_becomes_a_refusal(self):
        only = "Here's what I'd recommend:\n\n- Ibuprofen 200mg Tablets (fev-003) — x\n\nWhich one would you like to try?"
        reply, _ = ci.screen_reply(only, "I'm allergic to ibuprofen", CATALOG)
        assert reply.startswith("I can't recommend") and "allergy to ibuprofen" in reply

    def test_model_prose_gets_a_warning_appended(self):
        prose = "I recommend Pseudoephedrine Decongestant Tablets for you. Would you like to order this?"
        reply, excluded = ci.screen_reply(prose, "I have high blood pressure and a stuffy nose", CATALOG)
        assert reply.startswith(prose) and "Please be careful with:" in reply and excluded == {"col-002"}

    def test_replies_that_are_not_recommendations_are_untouched(self):
        status = "Order confirmed!\n\nProduct: Ibuprofen 200mg Tablets"
        assert ci.screen_reply(status, "I have a stomach ulcer", CATALOG) == (status, set())

    def test_no_stated_condition_changes_nothing(self):
        assert ci.screen_reply(FEVER_LIST, "I have a fever", CATALOG) == (FEVER_LIST, set())


@pytest.mark.usefixtures("isolated_db")
class TestStartOrder:
    def test_a_ruled_out_product_is_not_ordered(self):
        ci.set_current_text("I have a stomach ulcer, order Ibuprofen 200mg Tablets")
        result = json.loads(tools.start_order("fev-003", "s1"))
        assert "I can't set up an order for Ibuprofen 200mg Tablets" in result["error"]
        assert store.get_pending_order("s1") is None

    def test_a_condition_from_an_earlier_message_still_applies(self):
        store.save_session_messages("s1", [{"role": "user", "content": "I have kidney disease"}])
        ci.set_current_text("order Ibuprofen 200mg Tablets")
        assert "error" in json.loads(tools.start_order("fev-003", "s1"))

    def test_other_products_and_other_customers_are_unaffected(self):
        ci.set_current_text("I have a stomach ulcer")
        assert "error" not in json.loads(tools.start_order("fev-001", "s1"))
        ci.set_current_text("")
        assert "error" not in json.loads(tools.start_order("fev-003", "s2"))


@pytest.mark.usefixtures("isolated_db")
class TestRoute:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setattr(config, "SMTP_CONFIGURED", False)

        def _boom(*args, **kwargs):
            raise AssertionError("the chat model must not be reached")

        monkeypatch.setattr(agent, "create_agent", _boom)
        self.client = TestClient(main.app)

    def _chat(self, sid, text):
        res = self.client.post("/api/chat", json={"session_id": sid, "text": text})
        res.raise_for_status()
        return res.json()["reply"]

    @pytest.mark.parametrize("first, left_out", [
        ("I have a fever and stomach ulcer", "Ibuprofen 200mg Tablets"),
        ("I'm on warfarin and have a fever", "Ibuprofen 200mg Tablets"),
        ("I'm allergic to ibuprofen and have a fever", "Ibuprofen 200mg Tablets"),
        ("I'm pregnant and have a fever", "Ibuprofen 200mg Tablets"),
    ])
    def test_the_adult_fever_list_leaves_ibuprofen_out(self, first, left_out):
        assert "child, or for an adult" in self._chat("s1", first)
        reply = self._chat("s1", "for myself")
        assert "(fev-001)" in reply and "(fev-003)" not in reply.split("Left out")[0]
        assert left_out in reply.split("Left out")[1]

    def test_high_blood_pressure_leaves_the_decongestants_out_of_the_cold_list(self):
        self._chat("s1", "I have high blood pressure and a cold")
        reply = self._chat("s1", "adult")
        head, tail = reply.split("Left out")
        assert "(col-002)" not in head and "(col-003)" not in head and "(col-001)" in head
        assert "Pseudoephedrine" in tail and "Cold & Flu" in tail

    def test_a_ruled_out_product_cannot_be_picked_from_the_list_by_number(self):
        self._chat("s1", "I have a fever and stomach ulcer")
        self._chat("s1", "for myself")
        assert [p["id"] for p in store.get_last_products("s1")] == ["fev-001", "fev-002"]
        assert "address" in self._chat("s1", "2").lower()
        assert store.get_pending_order("s1")["product_id"] == "fev-002"

    def test_ordering_a_ruled_out_product_by_name_is_refused(self):
        reply = self._chat("s1", "I have a stomach ulcer, order Ibuprofen 200mg Tablets")
        assert "I can't set up an order" in reply and store.get_pending_order("s1") is None

    def test_without_a_stated_condition_nothing_changes(self):
        self._chat("s1", "I have a fever")
        reply = self._chat("s1", "for myself")
        assert "(fev-003)" in reply and "Left out" not in reply


class TestAQuestionIsNotAStatement:
    """Real bug (found by the harness on Nov 1): "can i take ibuprofen and
    paracetamol together" set the "taking another anti-inflammatory" condition, and
    a later order for Ibuprofen in the same session was refused with "you mentioned
    another anti-inflammatory"."""

    @pytest.mark.parametrize("text", [
        "can i take ibuprofen and paracetamol together", "is it safe to take aspirin with food", "should I take ibuprofen",
        "I want to take ibuprofen", "can I take paracetamol", "could I take naproxen", "do you take aspirin",
        "I wonder if I take ibuprofen with food",
    ])
    def test_a_question_or_intention_sets_no_condition(self, text):
        assert ci.conditions_in(text) == set()

    @pytest.mark.parametrize("text, key", [
        ("I take aspirin daily", "nsaid_use"), ("I am taking naproxen", "nsaid_use"), ("I am on ibuprofen", "nsaid_use"),
        ("I took paracetamol an hour ago", "paracetamol_use"), ("I'm taking paracetamol already", "paracetamol_use"),
    ])
    def test_a_statement_still_does(self, text, key):
        assert key in ci.conditions_in(text)

    def test_the_question_does_not_block_a_later_order_in_the_same_session(self):
        store.save_session_messages("s1", [{"role": "user", "content": "can i take ibuprofen and paracetamol together"}])
        ci.set_current_text("order Ibuprofen 200mg Tablets")
        assert "error" not in json.loads(tools.start_order("fev-003", "s1"))

    def test_a_real_statement_still_blocks_the_order(self):
        store.save_session_messages("s2", [{"role": "user", "content": "I take aspirin daily"}])
        ci.set_current_text("order Ibuprofen 200mg Tablets")
        assert "error" in json.loads(tools.start_order("fev-003", "s2"))


class TestAnAllergyStatementStaysInsideOneStatement:
    """Real bug (found by a harness case, Nov 3): the span read after "allergic"
    ran 60 characters across a newline into the NEXT message, so "I'm allergic to
    ibuprofen" followed by "order Paracetamol 500mg Tablets" refused the paracetamol
    order with "you mentioned an allergy to paracetamol"."""

    def test_the_next_message_is_not_part_of_the_allergy(self):
        text = "I'm allergic to ibuprofen\norder Paracetamol 500mg Tablets ship to 5 Mid Rd, Pune"
        assert set(ci.allergic_product_ids(text)) == {"fev-003"}

    @pytest.mark.parametrize("text", [
        "I'm allergic to ibuprofen but order paracetamol", "I'm allergic to ibuprofen, order Paracetamol",
        "I have an allergy to ibuprofen. Please order paracetamol", "allergic to ibuprofen\nI have a fever",
        "allergic to ibuprofen then buy paracetamol",
    ])
    def test_a_new_request_ends_the_allergy(self, text):
        assert set(ci.allergic_product_ids(text)) == {"fev-003"}

    @pytest.mark.parametrize("text, expected", [
        ("allergic to aspirin and paracetamol", {"fev-003", "fev-001", "fev-002", "fev-004", "col-003"}),
        ("I'm allergic to aspirin or ibuprofen", {"fev-003"}),
        ("paracetamol allergy", {"fev-001", "fev-002", "fev-004", "col-003"}),
    ])
    def test_a_real_list_of_allergies_still_counts_every_item(self, text, expected):
        assert set(ci.allergic_product_ids(text)) == expected

    def test_an_ibuprofen_allergy_in_one_message_does_not_block_paracetamol_in_the_next(self):
        store.save_session_messages("s1", [{"role": "user", "content": "I'm allergic to ibuprofen"}])
        ci.set_current_text("order Paracetamol 500mg Tablets")
        assert "error" not in json.loads(tools.start_order("fev-001", "s1"))
        ci.set_current_text("order Ibuprofen 200mg Tablets")
        assert "error" in json.loads(tools.start_order("fev-003", "s1"))


class TestANegationInOneMessageDoesNotCancelAConditionInTheNext:
    """Real bug (stateful fuzz, seed 77, Nov 3): "I can't breathe" followed by "I'm
    allergic to ibuprofen" read the allergy as negated ("can't ... allergic") because
    the 25-character lookback crossed the line break between the two messages, so an
    ibuprofen order went through."""

    @pytest.mark.parametrize("text, expected", [
        ("I can't breathe\nI'm allergic to ibuprofen", {"fev-003"}),
        ("I don't smoke\nI'm allergic to ibuprofen", {"fev-003"}),
        ("no, thanks\nI'm allergic to ibuprofen", {"fev-003"}),
    ])
    def test_an_allergy_after_a_negation_in_the_previous_message(self, text, expected):
        assert set(ci.allergic_product_ids(text)) == expected

    @pytest.mark.parametrize("text, key", [
        ("I don't smoke\nI have a stomach ulcer", "ulcer"), ("no history of ulcers\nI'm pregnant", "pregnant"),
        ("I have no kidney problems\nI have asthma", "asthma"), ("I can't breathe\nI'm on warfarin", "blood_thinner"),
    ])
    def test_a_condition_after_a_negation_in_the_previous_message(self, text, key):
        assert key in ci.conditions_in(text)

    @pytest.mark.parametrize("text", [
        "I'm not allergic to ibuprofen", "I have no kidney problems", "no history of ulcers", "I don't have asthma",
    ])
    def test_a_negation_inside_the_same_message_still_negates(self, text):
        assert ci.conditions_in(text) == set() and ci.allergic_product_ids(text) == {}

    def test_the_exact_sequence_from_the_fuzz_is_refused(self):
        store.save_session_messages("s1", [
            {"role": "user", "content": "I can't breathe"}, {"role": "assistant", "content": "..."},
            {"role": "user", "content": "I'm allergic to ibuprofen"}, {"role": "assistant", "content": "..."},
        ])
        ci.set_current_text("order Ibuprofen 200mg Tablets")
        assert "error" in json.loads(tools.start_order("fev-003", "s1"))


class TestTheAllergyDetectorIsFastOnHostileInput:
    """Real bug (found Nov 9 by timing every handler on hostile input): the "X
    allergy" pattern was QUADRATIC on a long run of spaces or letters -- 2.3
    seconds for one 4,000-character message (the API's maximum), and it runs several
    times per request over the whole conversation. A cheap way to stall the server."""

    @pytest.mark.parametrize("text", [" " * 4000, "a " * 2000, "x" * 4000, "allergy " * 500, "allergic to " * 300, "\n" * 4000, "- / " * 1000])
    def test_a_4000_character_message_is_scanned_in_well_under_a_second(self, text):
        import time

        started = time.perf_counter()
        ci.concerns(text)
        assert time.perf_counter() - started < 0.5   # the quadratic version took ~2.3 s on 4000 spaces

    def test_the_cost_does_not_grow_with_a_whole_long_conversation(self):
        import time

        history = "\n".join(["I have a fever and a cough and some other things to say " * 30] * 40)   # ~67,000 characters
        started = time.perf_counter()
        ci.concerns(history)
        assert time.perf_counter() - started < 1.0

    @pytest.mark.parametrize("text, expected", [
        ("paracetamol allergy", {"fev-001", "fev-002", "fev-004", "col-003"}),
        ("I have a paracetamol allergy", {"fev-001", "fev-002", "fev-004", "col-003"}),
        ("aspirin or paracetamol allergy", {"fev-003", "fev-001", "fev-002", "fev-004", "col-003"}),
        ("ibuprofen allergy", {"fev-003"}),
        ("allergy to ibuprofen", {"fev-003"}),
        ("I'm allergic to ibuprofen", {"fev-003"}),
        ("I can't breathe\nparacetamol allergy", {"fev-001", "fev-002", "fev-004", "col-003"}),
        ("I have a penicillin allergy and a fever", set()),
        ("not paracetamol allergy", set()),
        ("no ibuprofen allergy", set()),
        ("I don't have a paracetamol allergy", set()),
    ])
    def test_every_allergy_phrasing_still_gives_the_same_answer(self, text, expected):
        assert set(ci.allergic_product_ids(text)) == expected


class TestANegatedAllergyWithAnApostropheIsNotAnAllergy:
    """Real bug (found Nov 9; the code before today had it too): "I don't have a
    paracetamol allergy" counted as a paracetamol allergy and would block every
    paracetamol product. The apostrophe splits "don't" across the boundary of the
    text being matched, so the negation was lost."""

    @pytest.mark.parametrize("text", [
        "I don't have a paracetamol allergy", "I haven't got an ibuprofen allergy", "there's no ibuprofen allergy",
        "I do not have an ibuprofen allergy", "I'm not allergic to ibuprofen", "I don’t have a paracetamol allergy",
        "no, I don't have an ibuprofen allergy", "I don't think I have a paracetamol allergy",
    ])
    def test_it_is_not_an_allergy(self, text):
        assert ci.allergic_product_ids(text) == {}

    @pytest.mark.parametrize("text, expected", [
        ("I have a paracetamol allergy", {"fev-001", "fev-002", "fev-004", "col-003"}),
        ("I don't smoke\nI have a paracetamol allergy", {"fev-001", "fev-002", "fev-004", "col-003"}),
        ("I can't breathe\nI'm allergic to ibuprofen", {"fev-003"}),
        ("I don't know why but I have an ibuprofen allergy", {"fev-003"}),
    ])
    def test_a_real_allergy_after_an_unrelated_negation_still_counts(self, text, expected):
        assert set(ci.allergic_product_ids(text)) == expected

    def test_a_denied_allergy_does_not_block_the_order(self):
        store.save_session_messages("s1", [{"role": "user", "content": "I don't have a paracetamol allergy"}])
        ci.set_current_text("order Paracetamol 500mg Tablets")
        assert "error" not in json.loads(tools.start_order("fev-001", "s1"))


class TestPhrasingsAndBrandsThePatternsMissed:
    """Real gaps (Nov 10): "I have bad kidneys", "I'm a heart patient", "stomach
    bleeding", and brand names (Eliquis, Xarelto, Plavix, Coumadin, Ecosprin,
    Disprin, Voltaren, Combiflam) were not recognised, so an ibuprofen order
    for them went straight through."""

    IBUPROFEN = "fev-003"

    @staticmethod
    def _refusal(text, product_id="fev-003"):
        return ci.refusal_for(store.find_product(product_id), text)

    @pytest.mark.parametrize("text, label", [
        ("my kidneys are weak", "kidney problems"), ("I have bad kidneys", "kidney problems"),
        ("I'm a kidney patient", "kidney problems"), ("had a kidney transplant", "kidney problems"),
        ("I'm a heart patient", "a heart condition"), ("I have a weak heart", "a heart condition"),
        ("I had bypass surgery", "a heart condition"), ("I have a heart stent", "a heart condition"),
        ("stomach bleeding", "stomach bleeding"), ("I had gastric bleeding", "stomach bleeding"),
        ("history of GI bleed", "stomach bleeding"),
        ("I use eliquis", "a blood thinner"), ("I take Xarelto", "a blood thinner"),
        ("I take coumadin", "a blood thinner"), ("I take plavix", "a blood thinner"),
        ("I take blood thinning tablets", "a blood thinner"),
        ("I'm on ecosprin", "another anti-inflammatory"), ("taking disprin", "another anti-inflammatory"),
        ("I use voltaren gel", "another anti-inflammatory"), ("I take combiflam", "another anti-inflammatory"),
    ])
    def test_ibuprofen_is_refused_and_the_reason_is_named(self, text, label):
        refusal = self._refusal(text)
        assert refusal and label in refusal, (text, refusal)

    @pytest.mark.parametrize("text", [
        "I don't have bad kidneys", "no stomach bleeding", "I'm not a heart patient", "I take vitamin c",
        "what about kidney beans", "my heart is happy", "I don’t take plavix",
    ])
    def test_denials_and_unrelated_words_are_not_conditions(self, text):
        assert self._refusal(text) is None, text

    def test_combiflam_also_contains_paracetamol(self):
        assert self._refusal("I take combiflam", "fev-001")

    def test_a_heart_patient_is_also_kept_off_the_decongestant(self):
        assert self._refusal("I'm a heart patient", "col-002")


class TestIbuprofenAndAChild:
    """Real gap (Nov 12): the ibuprofen text gives an adult dose only, yet "order
    ibuprofen for my 6 month old" went through. The reply to "can I give ibuprofen
    to my 5 year old" also just switched to another product without answering."""

    @staticmethod
    def _refusal(text, product_id="fev-003"):
        return ci.refusal_for(store.find_product(product_id), text)

    @pytest.mark.parametrize("text", [
        "for my 5 year old (child)", "it's for my son (child)", "for a child", "for my toddler", "for the baby",
        "for my kids",
    ])
    def test_ibuprofen_is_refused_for_a_child(self, text):
        refusal = self._refusal(text)
        assert refusal and "a child" in refusal and "adult dose only" in refusal

    @pytest.mark.parametrize("text", ["not for a child, for me", "I'm an adult, not a kid", "for myself"])
    def test_an_adult_is_not_refused(self, text):
        assert self._refusal(text) is None

    def test_the_children_s_syrup_is_not_refused_for_a_child(self):
        assert self._refusal("for my 5 year old (child)", "fev-004") is None


class TestWhoItIsForCanChangeMidConversation:
    """Real bug (Nov 13, real model): after "fever" -> "child", asking "what about
    for adults?" got ibuprofen recommended AND the note "Please be careful with
    Ibuprofen: you mentioned a child". A later message that says adult / myself
    ends the child condition."""

    @staticmethod
    def _refusal(text):
        return ci.refusal_for(store.find_product("fev-003"), text)

    @pytest.mark.parametrize("text", [
        "my baby has a fever\nchild\nwhat about for adults?",
        "my baby has a fever\nactually for myself",
        "my son has a cold (child)\nfor me",
        "child\nadult",
    ])
    def test_a_later_adult_message_lifts_it(self, text):
        assert self._refusal(text) is None

    @pytest.mark.parametrize("text", [
        "adult\nchild",
        "my baby has a fever",
        "it's for my son (child)\norder ibuprofen",
        "my baby has a fever\nactually for myself\nno wait, it's for the baby",
        "child (child) and adult",          # one message mentioning both stays conservative
    ])
    def test_a_child_mentioned_last_or_alongside_still_blocks(self, text):
        assert self._refusal(text), text

    def test_the_sentence_that_screens_a_recommendation_follows_the_same_rule(self):
        catalog = store.get_catalog()
        reply = "For adults, Ibuprofen 200mg Tablets (fev-003) would be a good fit."
        screened, excluded = ci.screen_reply(reply, "fever\nchild\nwhat about for adults?", catalog)
        assert excluded == set() and "careful" not in screened.lower()


class TestAChildrensProductNameIsNotAChild:
    """Real bug in my Nov 12 change (found Nov 17 in a real browser): clicking "Ask
    Assistant" on Children's Paracetamol Syrup put the word "Children's" in the
    session, and an adult's later "order 2 ibuprofen" was refused with "you mentioned
    a child"."""

    @staticmethod
    def _refusal(text):
        return ci.refusal_for(store.find_product("fev-003"), text)

    @pytest.mark.parametrize("text", [
        "I'd like to order Children's Paracetamol Syrup\norder ibuprofen",
        "show me the Children’s Cold & Cough Syrup\norder ibuprofen",
        "do you have children's medicine\norder ibuprofen",
        "ibuprofen vs Children's Paracetamol Syrup",
        "what does the children's cough syrup cost",
    ])
    def test_the_product_name_alone_is_not_a_patient(self, text):
        assert self._refusal(text) is None, text

    @pytest.mark.parametrize("text", [
        "for my children", "for children", "my children have a fever", "is it safe for children",
        "children's paracetamol for my kid", "give children's syrup to my baby",
    ])
    def test_a_real_child_is_still_caught(self, text):
        assert self._refusal(text), text

    def test_main_uses_the_same_definition(self):
        from app import main
        assert main._ibuprofen_for_a_child_reply("ibuprofen vs Children's Paracetamol Syrup") is None
        assert main._ibuprofen_for_a_child_reply("can children take ibuprofen")


class TestAChildIsSeenEvenWhenTheAgeGlossWasSkipped:
    """Real bug (found Nov 17): while an order is pending the "(child)" gloss is skipped
    (an address must be stored exactly as typed), so "order ibuprofen for my 5 year old"
    typed at that moment was not recognised as a child, and the "yes" that followed
    ordered ibuprofen for a five-year-old."""

    @staticmethod
    def _refusal(text):
        return ci.refusal_for(store.find_product("fev-003"), text)

    @pytest.mark.parametrize("text", [
        "order ibuprofen for my 5 year old", "for my 6 months old", "it is for my 8yo", "for my son",
        "order ibuprofen for my daughter", "I'm 5",
    ])
    def test_a_stated_child_age_or_my_son_blocks_ibuprofen_without_the_gloss(self, text):
        assert self._refusal(text), text

    @pytest.mark.parametrize("text", [
        "I'm 30, order ibuprofen", "my son is 4 and I'm 35", "for my husband", "12 Baker Street, London",
        "order ibuprofen for my 14 year old", "7 Yo Street, Leeds", "I'm 5 minutes from the shop",
        "my 5 year old has a fever\nactually for me, I'm an adult",
    ])
    def test_adults_addresses_and_teenagers_are_not_blocked(self, text):
        assert self._refusal(text) is None, text


LABEL_PHOTO = (
    "what is this\n[Image analysis: Ibuprofen 200mg tablets. Warnings: not recommended for people with a history of "
    "stomach ulcers, kidney problems, or heart conditions. Pregnant women should avoid ibuprofen. Keep away from "
    "children. Do not take with blood thinners or other NSAIDs.]"
)


class TestAPhotoOfALabelIsNotTheCustomersMedicalHistory:
    """Real bug (found Nov 18): the photo-reading step's "[Image analysis: ...]" is stored in
    the history as part of the user's message. A photo of an ibuprofen box carries the
    box's own warnings (ulcers, kidney, heart, pregnancy, children, blood thinners), which
    were read as the customer's conditions: ordering the medicine they had just
    photographed was refused with "you mentioned kidney problems"."""

    @staticmethod
    def _refusal(texts, product_id="fev-003"):
        return ci.refusal_for(store.find_product(product_id), ci.stated_text(texts))

    def test_the_label_text_states_no_condition(self):
        assert ci.conditions_in(ci.stated_text([LABEL_PHOTO])) == set()

    @pytest.mark.parametrize("product_id", ["fev-001", "fev-003", "col-002", "col-003"])
    def test_no_product_is_refused_because_of_a_label_photo(self, product_id):
        assert self._refusal([LABEL_PHOTO], product_id) is None

    def test_what_the_customer_really_says_is_still_counted(self):
        assert "stomach ulcer" in self._refusal([LABEL_PHOTO, "I have a stomach ulcer"])
        assert "kidney problems" in self._refusal(["I have kidney disease, what is this\n[Image analysis: Ibuprofen 200mg tablets.]"])

    def test_a_multi_line_analysis_is_removed_whole(self):
        text = ci.stated_text(["x\n[Image analysis: line one\nkidney problems on line two]\nI am 30"])
        assert "kidney" not in text and "I am 30" in text

    def test_start_order_goes_through_after_a_label_photo(self):
        store.save_session_messages("s-photo", [{"role": "user", "content": LABEL_PHOTO}])
        ci.set_current_text("order Ibuprofen 200mg Tablets")
        assert "error" not in json.loads(tools.start_order("fev-003", "s-photo"))

    def test_start_order_is_still_refused_when_the_customer_states_a_condition_after_the_photo(self):
        store.save_session_messages("s-photo2", [{"role": "user", "content": LABEL_PHOTO}, {"role": "user", "content": "I have a stomach ulcer"}])
        ci.set_current_text("order Ibuprofen 200mg Tablets")
        assert "error" in json.loads(tools.start_order("fev-003", "s-photo2"))

    def test_the_completion_check_ignores_the_label_too(self):
        from app import main
        store.save_session_messages("s-photo3", [{"role": "user", "content": LABEL_PHOTO}])
        ci.set_current_text("12 Baker Street, London")
        assert main._refusal_at_completion("s-photo3", "fev-003") is None


class TestInjectedModelHintsAreNotTheCustomersWords:
    """The same fault through a second door (found Nov 18): the "[Detected a ... Matching
    catalog products: ...]" hint stored in the user message lists product descriptions,
    and the children's syrups' "formulated for children" made the child condition true for
    the rest of the session."""

    HINT = (
        "I have a fever, what should I take\n[Detected a fever-related medicine/symptom mention. Matching catalog "
        "products (don't call lookup_symptom again): {\"matched\": true, \"products\": [{\"id\": \"fev-004\", \"name\": "
        "\"Children's Paracetamol Syrup\", \"description\": \"Liquid fever reducer formulated for children.\"}]}. "
        "Present these as a recommendation and ask if they'd like to order one — do NOT call start_order yet unless "
        "this message already contains clear ordering language (e.g. 'order X', 'buy X'); wait for their next "
        "message to confirm.]"
    )

    def test_the_hint_block_states_no_condition(self):
        assert ci.conditions_in(ci.stated_text([self.HINT])) == set()

    def test_ibuprofen_is_not_refused_after_such_a_turn(self):
        assert ci.refusal_for(store.find_product("fev-003"), ci.stated_text([self.HINT])) is None

    def test_the_clarify_note_and_the_unreadable_photo_note_are_removed_too(self):
        text = ci.stated_text(["my kid\n[Clarify: is this for a child or an adult?]", "[Photo attached — could not be read]"])
        assert "Clarify" not in text and "Photo attached" not in text and "my kid" in text

    def test_the_customers_own_words_in_the_same_message_still_count(self):
        assert "a child" in ci.refusal_for(
            store.find_product("fev-003"), ci.stated_text(["it's for my baby\n" + self.HINT.split("\n", 1)[1]])
        )
