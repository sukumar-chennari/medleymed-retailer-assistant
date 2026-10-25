"""Combination questions are answered from the product warnings (app/interactions.py).

Real bug (Oct 22 probe): "can I take cetirizine and pseudoephedrine together" and
"can I take ibuprofen and paracetamol together" were answered "yes" from the model's
general knowledge, while the knowledge base has specific warnings (never double up on
paracetamol, never combine two decongestants, ...) and says nothing about those pairs."""

import pytest
from fastapi.testclient import TestClient

from app import config, interactions as it, main, store

CATALOG = store.get_catalog()
client = TestClient(main.app)


class TestWarnedCombinations:
    @pytest.mark.parametrize("text, both", [
        ("can I take paracetamol and Cold & Flu Multi-Symptom Relief together", "paracetamol"),
        ("can I take Paracetamol 500mg Tablets and Paracetamol Extra Strength 650mg together", "paracetamol"),
        ("can I take Children's Paracetamol Syrup with Paracetamol 500mg Tablets", "paracetamol"),
        ("can I take pseudoephedrine with Cold & Flu Multi-Symptom Relief", "a decongestant"),
        ("can I take Cough Suppressant Syrup (Dextromethorphan) and Children's Cold & Cough Syrup together", "dextromethorphan"),
    ])
    def test_a_shared_ingredient_gets_the_knowledge_bases_warning(self, text, both):
        reply = it.combination_reply(text, CATALOG)
        assert reply.startswith("No —") and f"both contain {both}" in reply and "warns:" in reply

    def test_the_childrens_syrup_with_an_antihistamine_is_warned_against(self):
        reply = it.combination_reply("can I give Children's Cold & Cough Syrup with cetirizine", CATALOG)
        assert reply.startswith("No — for Children's Cold & Cough Syrup") and "antihistamine or cough suppressant" in reply

    @pytest.mark.parametrize("text", [
        "can I take cetirizine and pseudoephedrine together", "can I take ibuprofen and paracetamol together",
        "is it ok to take cetirizine with guaifenesin", "can I take paracetamol 500mg and ibuprofen 200mg together",
    ])
    def test_a_pair_the_knowledge_base_is_silent_on_gets_an_honest_not_covered(self, text):
        reply = it.combination_reply(text, CATALOG)
        assert "doesn't say whether" in reply and "ask a pharmacist or doctor" in reply
        assert not reply.startswith(("Yes", "No"))


class TestBrandNamesAndNsaids:
    """Brand names resolve to their active ingredient, and aspirin/naproxen are
    NSAIDs, which the knowledge base says ibuprofen must not be combined with."""

    @pytest.mark.parametrize("text, both", [
        ("can I take Tylenol with Cold & Flu Multi-Symptom Relief", "paracetamol"),
        ("can I take Crocin and Dolo together", "paracetamol"),
        ("can I take Tylenol and paracetamol together", "paracetamol"),
        ("can I take Sudafed with Cold & Flu Multi-Symptom Relief", "a decongestant"),
        ("can I take aspirin and ibuprofen together", "an anti-inflammatory (NSAID)"),
        ("can I take Advil and aspirin together", "an anti-inflammatory (NSAID)"),
        ("can I take naproxen with Ibuprofen 200mg Tablets", "an anti-inflammatory (NSAID)"),
    ])
    def test_a_shared_ingredient_behind_brand_names_gets_the_warning(self, text, both):
        reply = it.combination_reply(text, CATALOG)
        assert reply.startswith("No —") and f"both contain {both}" in reply and "warns:" in reply

    @pytest.mark.parametrize("text", [
        "can I take Tylenol and Advil together", "can I take Advil and Zyrtec together",
        "can I take Tylenol and aspirin together", "can I take Tylenol and vitamin C together",
    ])
    def test_a_pair_with_no_shared_ingredient_gets_the_honest_not_covered(self, text):
        reply = it.combination_reply(text, CATALOG)
        assert "doesn't say whether" in reply and not reply.startswith(("Yes", "No"))

    def test_the_brand_is_shown_with_its_ingredient(self):
        reply = it.combination_reply("can I take Tylenol and Advil together", CATALOG)
        assert "Tylenol (paracetamol)" in reply and "Advil (ibuprofen)" in reply

    @pytest.mark.parametrize("text", ["can I take Tylenol", "what is Tylenol", "do you have Tylenol and Advil"])
    def test_single_brand_and_availability_questions_are_left_alone(self, text):
        assert it.combination_reply(text, CATALOG) is None


class TestNotCombinationQuestions:
    @pytest.mark.parametrize("text", [
        "what is dextromethorphan", "can I take ibuprofen", "order paracetamol and ibuprofen",
        "how much paracetamol and ibuprofen can I take together", "should I take paracetamol with food",
        "can I take cetirizine for cold and cough", "price of paracetamol and ibuprofen",
        "I have a fever and a cold", "",
    ])
    def test_these_are_left_alone(self, text):
        assert it.combination_reply(text, CATALOG) is None


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

    def test_the_warning_is_answered_without_the_model(self):
        reply = self._chat("s1", "can I take paracetamol and Cold & Flu Multi-Symptom Relief together")
        assert reply.startswith("No —") and "accidental overdose" in reply

    def test_an_order_awaiting_its_address_is_left_alone(self, monkeypatch):
        store.set_pending_order("s1", "fev-001", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert self._chat("s1", "can I take paracetamol and ibuprofen together") == "from the model"
        assert store.get_pending_order("s1")["product_id"] == "fev-001"

    def test_brand_combination_questions_are_answered_without_the_model(self):
        assert self._chat("s1", "can I take Tylenol with Cold & Flu Multi-Symptom Relief").startswith("No —")
        assert "doesn't say whether" in self._chat("s2", "can I take Tylenol and Advil together")
