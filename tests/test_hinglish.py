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


@pytest.mark.usefixtures("isolated_db")
class TestTheGlossNeverTouchesAnAddress:
    """Real bug of my own (found Oct 26): the gloss ran on every message, so an
    address containing such a word — "12 Khansi Road, Bachha Nagar" — was saved
    as "12 Khansi Road, Bachha Nagar (cough; child)"."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setattr(config, "SMTP_CONFIGURED", False)
        self.client = client

    def _chat(self, sid, text):
        res = self.client.post("/api/chat", json={"session_id": sid, "text": text})
        res.raise_for_status()
        return res.json()["reply"]

    def test_a_first_time_address_is_saved_exactly_as_typed(self):
        store.set_pending_order("s1", "fev-003", 1)
        reply = self._chat("s1", "12 Khansi Road, Bachha Nagar, Hyderabad")
        assert "Shipping to: 12 Khansi Road, Bachha Nagar, Hyderabad\n" in reply + "\n"
        assert "(cough" not in reply and store.get_address("demo_user") == "12 Khansi Road, Bachha Nagar, Hyderabad"

    def test_a_different_address_at_the_confirmation_step_is_saved_exactly_as_typed(self):
        store.save_address("demo_user", "10 High St, Leeds")
        store.set_pending_address_confirmation("s1", "fev-003", 1)
        self._chat("s1", "7 Zukam Lane, Sardi Colony, Pune")
        assert store.get_address("demo_user") == "7 Zukam Lane, Sardi Colony, Pune"

    def test_an_email_is_never_glossed(self):
        order = store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)
        store.set_pending_email("s1", order["order_id"])
        self._chat("s1", "sardi@example.com")
        assert store.get_email("demo_user") == "sardi@example.com"

    def test_a_hinglish_symptom_message_is_still_glossed(self, monkeypatch):
        from app import agent

        monkeypatch.setattr(agent, "create_agent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no model")))
        assert "child, or for an adult" in self._chat("s1", "mujhe bukhar hai")
