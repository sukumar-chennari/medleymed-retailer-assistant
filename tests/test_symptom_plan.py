"""A cough together with a fever and/or cold symptoms is planned as a whole.

Real bugs (Oct 28 probe): "I'm an adult with a fever and a dry cough" and "... a
wet cough, runny nose and fever" were answered with the cough product ONLY — the
fever and the runny nose were silently dropped; "I have a fever, cough, and runny
nose" asked "dry or chesty?" and the natural reply "for myself" was declined as
out of scope."""

import pytest
from fastapi.testclient import TestClient

from app import agent, config, main, store, symptom_plan

client = TestClient(main.app)


@pytest.mark.usefixtures("isolated_db")
class TestGroups:
    @pytest.mark.parametrize("text, groups", [
        ("I have a fever, cough, and runny nose", ["fever", "cough", "cold"]),
        ("I'm an adult with a fever and a dry cough", ["fever", "cough"]),
        ("a wet cough, runny nose and fever", ["fever", "cough", "cold"]),
        ("cough and sore throat", ["cough", "cold"]),
        ("I have a dry cough", ["cough"]),
        ("I have a fever", ["fever"]),
        ("runny nose and sneezing", ["cold"]),
    ])
    def test_the_groups_named(self, text, groups):
        assert symptom_plan.groups_in(text) == groups

    @pytest.mark.parametrize("text", ["I have a dry cough", "I have a fever", "runny nose and sneezing", "my cough is wet", "hello"])
    def test_a_single_group_is_left_to_the_existing_logic(self, text):
        assert symptom_plan.start(text, "s1") is None
        assert store.get_pending_clarification("s1") is None

    def test_detecting_age_and_cough_type(self):
        assert symptom_plan.detect_age("my child (child)") == "child" and symptom_plan.detect_age("for myself") == "adult"
        assert symptom_plan.detect_age("child and adult") == "" and symptom_plan.detect_age("no idea") == ""
        assert symptom_plan.detect_cough_type("dry") == "dry" and symptom_plan.detect_cough_type("chesty") == "wet"
        assert symptom_plan.detect_cough_type("dry and wet") == ""


@pytest.mark.usefixtures("isolated_db")
class TestRoute:
    @pytest.fixture(autouse=True)
    def _no_llm(self, monkeypatch):
        monkeypatch.setattr(config, "SMTP_CONFIGURED", False)

        def _boom(*args, **kwargs):
            raise AssertionError("the chat model must not be reached")

        monkeypatch.setattr(agent, "create_agent", _boom)

    def _chat(self, sid, text):
        res = client.post("/api/chat", json={"session_id": sid, "text": text})
        res.raise_for_status()
        return res.json()["reply"]

    @staticmethod
    def _ids(reply):
        import re
        return sorted(set(re.findall(r"(?:fev|col)-\d{3}", reply)))

    def test_a_dry_cough_with_a_fever_recommends_both_not_just_the_cough(self):
        reply = self._chat("s1", "I'm an adult with a fever and a dry cough")
        assert self._ids(reply) == ["col-004", "fev-001", "fev-002", "fev-003"]
        assert "For the fever:" in reply and "For the dry cough:" in reply

    def test_a_wet_cough_with_cold_symptoms_and_a_fever_recommends_all_three_groups(self):
        reply = self._chat("s1", "I'm an adult with a wet cough, runny nose and fever")
        assert self._ids(reply) == ["col-001", "col-002", "col-005", "fev-001", "fev-002", "fev-003"]
        assert "For the cold symptoms" in reply

    def test_cold_and_flu_is_left_out_when_fever_products_are_offered(self):
        # it contains paracetamol; the knowledge base says not to double up
        reply = self._chat("s1", "I'm an adult with a wet cough, runny nose and fever")
        assert "col-003" not in reply
        assert "col-003" in self._chat("s2", "I'm an adult with a dry cough and runny nose")

    def test_the_missing_pieces_are_asked_together_then_one_at_a_time(self):
        first = self._chat("s1", "I have a fever, cough, and runny nose")
        assert first.startswith("Two quick questions") and "child" in first and "dry" in first
        second = self._chat("s1", "for myself")       # the reply that used to be declined
        assert "dry" in second and "Two quick" not in second
        final = self._chat("s1", "dry")
        assert self._ids(final) == ["col-001", "col-002", "col-004", "fev-001", "fev-002", "fev-003"]

    def test_both_answers_in_one_reply_finish_it(self):
        self._chat("s1", "I have a fever, cough, and runny nose")
        final = self._chat("s1", "it's for myself and the cough is chesty")
        assert "col-005" in final and "fev-001" in final

    def test_age_given_up_front_only_asks_about_the_cough(self):
        reply = self._chat("s1", "I have a fever, cough and runny nose, I'm an adult")
        assert "dry" in reply and "child, or for an adult" not in reply

    def test_a_child_gets_only_childrens_products(self):
        reply = self._chat("s1", "my 4 year old has a fever, dry cough and runny nose")
        assert self._ids(reply) == ["col-006", "fev-004"]

    def test_a_childs_wet_cough_has_no_product_and_says_so(self):
        self._chat("s1", "my 4 year old has a fever, cough and runny nose")
        reply = self._chat("s1", "wet")
        assert self._ids(reply) == ["fev-004"] and "pharmacist or pediatrician" in reply

    def test_the_shown_products_can_be_picked_and_ordered(self):
        self._chat("s1", "I'm an adult with a fever and a dry cough")
        assert [p["id"] for p in store.get_last_products("s1")] == ["fev-001", "fev-002", "fev-003", "col-004"]
        self._chat("s1", "4")   # a bare number picks from the list just shown
        assert store.get_pending_order("s1")["product_id"] == "col-004"

    def test_the_condition_screen_still_applies(self):
        reply = self._chat("s1", "I have a stomach ulcer, I'm an adult with a fever and a dry cough")
        assert "(fev-003)" not in reply.split("Left out")[0] and "Left out because of what you told me" in reply

    def test_an_info_question_is_not_planned(self):
        # It reaches the (stubbed, failing) chat model instead of a plan: the
        # generic failure reply, never a symptom-by-symptom recommendation.
        reply = self._chat("s1", "what are the side effects of cough syrup if I also have a fever")
        assert "by symptom" not in reply and reply == main.GENERIC_FAILURE_REPLY

    def test_a_single_symptom_message_keeps_its_existing_flow(self):
        assert "child, or for an adult" in self._chat("s1", "I have a fever")
        assert "dry" in self._chat("s2", "I have a cough").lower()
