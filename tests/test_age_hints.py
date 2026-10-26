"""Who the medicine is for, when the message already says (app/age_hints.py).

Real gap (Oct 26 probe): 14 of 16 messages that state it — "my 3 year old has a
fever", "my son has a cold", "my wife has a fever", "I'm 25 and have a fever" —
were still asked "child or adult?", and "my 5 year old has a dry cough" never
reached the children's product."""

import pytest
from fastapi.testclient import TestClient

from app import age_hints, agent, config, main, store

client = TestClient(main.app)


class TestAgeGroup:
    @pytest.mark.parametrize("text", [
        "my 3 year old has a fever", "my son has a cold", "my daughter has a cough", "my 10 year old has a runny nose",
        "my 8 month old has a fever", "my 5 year old has a dry cough", "my three year old has a fever",
        "my 6-week-old has a fever", "give me something for my son", "my 5yo has a cold", "my newborn is sneezing",
    ])
    def test_child(self, text):
        assert age_hints.age_group(text) == "child"
        assert age_hints.gloss(text) == f"{text} (child)"

    @pytest.mark.parametrize("text", [
        "my grandmother has a fever", "I'm 25 and have a fever", "my 70 year old father has a cold",
        "my wife has a fever", "my husband has a wet cough", "for my mother, she has a fever", "I am 40 and have a cold",
        "my 25 year old son has a fever",  # an explicit age beats the relation
    ])
    def test_adult(self, text):
        assert age_hints.age_group(text) == "adult"

    @pytest.mark.parametrize("text", [
        "my teenage son has a fever", "I'm 16 and I have a cold", "my 14 year old has a fever",   # teens: ask
        "my son and I both have a cold", "the whole family has a fever", "my wife and my son have a fever",
        "I'm 30 and my 4 year old has a cold", "my 70 year old father and 5 year old son have a cold",  # mixed
        "my baby has a cold", "my toddler has a fever", "for my child", "I have a fever", "",  # already clear / unknown
        "I'm taking 500 mg", "I have a fever for 3 days", "I'm 5 days into a cold", "Take 2 tablets",
    ])
    def test_nothing_is_appended_when_it_is_unclear_or_already_said(self, text):
        assert age_hints.age_group(text) is None
        assert age_hints.gloss(text) == text


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

    @pytest.mark.parametrize("text, expected", [
        ("my 3 year old has a fever", "Children's Paracetamol Syrup"),
        ("my 8 month old has a fever", "Children's Paracetamol Syrup"),
        ("my son has a cold", "Children's Cold & Cough Syrup"),
        ("my 5 year old has a dry cough", "Children's Cold & Cough Syrup"),
        ("my wife has a fever", "Paracetamol 500mg Tablets (fev-001)"),
        ("I'm 25 and have a fever", "Paracetamol 500mg Tablets (fev-001)"),
        ("my 70 year old father has a cold", "Cetirizine 10mg Antihistamine Tablets (col-001)"),
        ("my husband has a wet cough", "Guaifenesin Expectorant Syrup"),
    ])
    def test_a_stated_age_goes_straight_to_the_right_products(self, text, expected):
        reply = self._chat("s1", text)
        assert expected in reply and "child, or for an adult" not in reply

    def test_a_childs_fever_never_gets_the_adult_products(self):
        reply = self._chat("s1", "my 3 year old has a fever")
        assert "fev-001" not in reply and "Ibuprofen" not in reply

    @pytest.mark.parametrize("text", [
        "my teenage son has a fever", "I'm 16 and I have a cold", "my son and I both have a cold",
    ])
    def test_an_unclear_message_is_still_asked(self, text):
        assert "child, or for an adult" in self._chat("s1", text)

    def test_a_daughters_cough_still_asks_dry_or_wet_and_then_gives_the_childrens_syrup(self):
        assert "dry" in self._chat("s1", "my daughter has a cough").lower()
        assert "Children's Cold & Cough Syrup" in self._chat("s1", "dry")

    @pytest.mark.parametrize("address", ["Mother Teresa Road, Hyderabad", "12 Son Street, Pune", "Wife Lane 4, Delhi"])
    def test_an_address_is_never_altered(self, address):
        store.set_pending_order("s1", "fev-003", 1)
        reply = self._chat("s1", address)
        assert f"Shipping to: {address}\n" in reply + "\n" and "(adult)" not in reply and "(child)" not in reply
        assert store.get_address("demo_user") == address

    def test_a_stated_age_works_as_the_answer_to_the_question_too(self):
        self._chat("s1", "I have a fever")
        assert "Children's Paracetamol Syrup" in self._chat("s1", "it's for my 3 year old")
