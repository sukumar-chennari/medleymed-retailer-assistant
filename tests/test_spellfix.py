"""Conservative spelling normalisation (app/spellfix.py).

Real bugs (Oct 31/Nov 1 probes): "how much is ibuprofin" and "order ibuprofin
200mg tablets" got the child-or-adult question; "ordr Ibuprofen 200mg Tablets"
and "order Ibuprofen 200 mg Tablets" were not recognised; and "can i take
ibuprofen and paracetmol together" got the model's own "Yes, you can take ..."
because the misspelling defeated the combination-warning logic."""

import re

import pytest
from fastapi.testclient import TestClient

from app import agent, config, data_ingest, main, spellfix, store

client = TestClient(main.app)


class TestCorrect:
    @pytest.mark.parametrize("text, expected", [
        ("how much is ibuprofin", "how much is ibuprofen"),
        ("price of paracetmol", "price of paracetamol"),
        ("I want to buy cetrizine", "I want to buy cetirizine"),
        ("do you have pseudoephedrin", "do you have pseudoephedrine"),
        ("what is dextromethorfan", "what is dextromethorphan"),
        ("Ibuprofin 200mg", "Ibuprofen 200mg"),                 # case is kept
        ("IBUPROFIN", "IBUPROFEN"),
        ("order Ibuprofen 200 mg Tablets", "order Ibuprofen 200mg Tablets"),
        ("take 10 ml of it", "take 10ml of it"),
        ("ordr Ibuprofen 200mg Tablets", "order Ibuprofen 200mg Tablets"),
        ("gimme ibuprofen", "give me ibuprofen"),
        ("can i take ibuprofen and paracetmol together", "can i take ibuprofen and paracetamol together"),
    ])
    def test_corrections(self, text, expected):
        assert spellfix.correct(text) == expected

    @pytest.mark.parametrize("text", [
        "I have a fever", "ibuprofen", "paracetamol", "antihistamines", "decongestants", "symptoms of a cold",
        "Tylonol", "sukumar@example.com ibuprofin", "12 High Street, Leeds", "", "hello there",
    ])
    def test_correct_or_unknown_text_is_untouched(self, text):
        assert spellfix.correct(text) == text

    def test_no_ordinary_word_is_rewritten_into_a_drug_name(self):
        # Every 7+ letter token in the knowledge base documents, plus common words
        # a customer might type: none may be changed by the corrector.
        corpus = " ".join(p.read_text(encoding="utf-8") for p in data_ingest.KB_DIR.glob("*.md"))
        common = (
            "yesterday tomorrow medicine medicines pharmacy pharmacist question symptoms headache fatigue shipping delivery "
            "address confirm confirmation recommend recommended allergic allergy breathing swallowing chemist different "
            "instructions available including antibiotic vitamin painkiller children's whatever something everything "
            "anything hospital emergency appointment ambulance temperature thermometer stomach discomfort nausea vomiting "
            "diarrhea constipation dizziness drowsiness infection prescription ingredient expiry capsules lozenges"
        ).split()
        changed = {
            (w, spellfix.correct(w))
            for w in set(re.findall(r"[A-Za-z]{7,}", corpus)) | set(common)
            if spellfix.correct(w).lower() != w.lower()
        }
        assert changed == set()


@pytest.mark.usefixtures("isolated_db")
class TestRoute:
    @pytest.fixture(autouse=True)
    def _no_model(self, monkeypatch):
        monkeypatch.setattr(config, "SMTP_CONFIGURED", False)

        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    def _chat(self, sid, text):
        res = client.post("/api/chat", json={"session_id": sid, "text": text})
        res.raise_for_status()
        return res.json()["reply"]

    @pytest.mark.parametrize("text, expected", [
        ("how much is ibuprofin", "Ibuprofen 200mg Tablets costs $5.29."),
        ("how much does cetirizine 10 mg cost", "Cetirizine 10mg Antihistamine Tablets costs $5.99."),
    ])
    def test_prices_despite_the_typo(self, text, expected):
        assert self._chat("s1", text) == expected

    @pytest.mark.parametrize("text, product_id", [
        ("order ibuprofin 200mg tablets", "fev-003"), ("I want to buy cetrizine", "col-001"),
        ("ordr Ibuprofen 200mg Tablets", "fev-003"), ("order Ibuprofen 200 mg Tablets", "fev-003"),
        ("gimme ibuprofen 200mg tablets", "fev-003"),
    ])
    def test_orders_despite_the_typo(self, text, product_id):
        reply = self._chat("s1", text)
        assert "address" in reply.lower() and store.get_pending_order("s1")["product_id"] == product_id

    def test_the_combination_warning_survives_a_misspelling(self):
        reply = self._chat("s1", "can i take ibuprofen and paracetmol together")
        assert "doesn't say whether" in reply and not reply.startswith(("Yes", "No"))
        assert self._chat("s2", "can I take Tylenol with Cold & Flu Multi-Symptom Relief").startswith("No —")

    def test_an_address_is_never_spell_corrected(self):
        store.set_pending_order("s1", "fev-003", 1)
        reply = self._chat("s1", "12 Ibuprofin Road, Pune")
        assert "Shipping to: 12 Ibuprofin Road, Pune" in reply and store.get_address("demo_user") == "12 Ibuprofin Road, Pune"

    def test_an_email_is_never_touched(self):
        order = store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)
        store.set_pending_email("s1", order["order_id"])
        self._chat("s1", "ibuprofin@example.com")
        assert store.get_email("demo_user") == "ibuprofin@example.com"
