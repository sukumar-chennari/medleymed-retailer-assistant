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
