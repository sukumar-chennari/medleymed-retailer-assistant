"""Romanised Hindi symptom words (app/hinglish.py).

Real bug (Oct 17 live probe): "mujhe bukhar hai" ("I have a fever") and "mera
bachha ko khansi hai" ("my child has a cough") were declined as out of scope,
because the symptom vocabulary is English-only."""

import pytest
from fastapi.testclient import TestClient

from app import agent, config, hinglish, main, store

client = TestClient(main.app)


class TestGloss:
    @pytest.mark.parametrize("text, expected", [
        ("mujhe bukhar hai", "mujhe bukhar hai (fever)"),
        ("mere bacche ko tez bukhaar hai", "mere bacche ko tez bukhaar hai (fever; child)"),
        ("mera bachha ko khansi hai", "mera bachha ko khansi hai (cough; child)"),
        ("sardi aur zukam hai", "sardi aur zukam hai (cold)"),
        ("gale mein dard hai", "gale mein dard hai (sore throat)"),
        ("naak beh rahi hai aur chheenk aa rahi hai", "naak beh rahi hai aur chheenk aa rahi hai (runny nose; sneezing)"),
        ("sir dard aur badan dard", "sir dard aur badan dard (headache; body ache)"),
        ("sukhi khansi hai, khud ke liye", "sukhi khansi hai, khud ke liye (cough; for myself; dry)"),
        ("balgam wali khansi", "balgam wali khansi (cough; wet cough with phlegm)"),
    ])
    def test_hinglish_words_get_their_english_meaning_appended(self, text, expected):
        assert hinglish.gloss(text) == expected

    @pytest.mark.parametrize("word", ["bachha", "bacha", "baccha", "bachcha", "bacche", "bachche", "bachchi"])
    def test_the_common_spellings_of_child_are_all_recognised(self, word):
        assert hinglish.gloss(word) == f"{word} (child)"

    @pytest.mark.parametrize("text", [
        "I have a fever", "I have a cold", "my baby has a cough", "order Ibuprofen 200mg Tablets", "",
        "mere bachhe ke liye dawai chahiye (child)", "the bach choir sings", "I need a sardine sandwich",
    ])
    def test_english_text_and_already_glossed_text_are_unchanged(self, text):
        assert hinglish.gloss(text) == text

    def test_for_my_child_is_not_read_as_for_myself(self):
        assert "for myself" not in hinglish.gloss("mere bachhe ke liye dawai chahiye")

    def test_an_english_word_already_present_is_not_repeated(self):
        assert hinglish.gloss("bukhar fever") == "bukhar fever"


@pytest.mark.usefixtures("isolated_db")
class TestRoute:
    @pytest.fixture(autouse=True)
    def _mock_email(self, monkeypatch):
        monkeypatch.setattr(config, "SMTP_CONFIGURED", False)

    def _chat(self, sid, text):
        res = client.post("/api/chat", json={"session_id": sid, "text": text})
        res.raise_for_status()
        return res.json()["reply"]

    @staticmethod
    def _no_llm(monkeypatch):
        # The clarifying question is deterministic inside run_turn; only the
        # chat model itself must stay out of it.
        def _boom(*args, **kwargs):
            raise AssertionError("the chat model must not be reached")

        monkeypatch.setattr(agent, "create_agent", _boom)

    def test_a_hinglish_fever_gets_the_clarifying_question_not_the_decline(self, monkeypatch):
        self._no_llm(monkeypatch)
        reply = self._chat("s1", "mujhe bukhar hai")
        assert "child, or for an adult" in reply  # the deterministic clarification, no model needed

    def test_answering_for_myself_then_recommends_adult_fever_products(self, monkeypatch):
        self._no_llm(monkeypatch)
        self._chat("s1", "mujhe bukhar hai")
        reply = self._chat("s1", "khud ke liye")
        assert "fev-001" in reply and "fev-004" not in reply

    def test_a_childs_hinglish_cough_asks_dry_or_wet(self, monkeypatch):
        self._no_llm(monkeypatch)
        assert "dry" in self._chat("s1", "mera bachha ko khansi hai").lower()

    def test_a_hinglish_emergency_phrase_in_english_still_escalates_first(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert "emergency" in self._chat("s1", "mere bacche ko bukhar hai, he can't breathe").lower()
