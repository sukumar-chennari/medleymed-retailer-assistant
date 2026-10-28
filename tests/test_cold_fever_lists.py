"""Adult cold and fever+cold recommendations only contain relevant products.

Real bug (Oct 28 probe): an adult with a fever plus a sore throat or stuffy nose
was shown all 8 products — both cough syrups (no cough was mentioned) and Cold &
Flu Multi-Symptom Relief next to the paracetamol fever tablets, which the
knowledge base says not to double up on. An adult with only a sore throat got the
two cough syrups too."""

import re

import pytest
from fastapi.testclient import TestClient

from app import agent, config, main

client = TestClient(main.app)
pytestmark = pytest.mark.usefixtures("isolated_db")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setattr(config, "SMTP_CONFIGURED", False)

    def _boom(*args, **kwargs):
        raise AssertionError("the chat model must not be reached")

    monkeypatch.setattr(agent, "create_agent", _boom)


def _ids(text):
    res = client.post("/api/chat", json={"session_id": "s1", "text": text})
    res.raise_for_status()
    return sorted(set(re.findall(r"(?:fev|col)-\d{3}", res.json()["reply"])))


@pytest.mark.parametrize("text", [
    "I have a sore throat and a fever, I'm an adult", "I'm an adult with a headache and a stuffy nose",
    "I'm an adult with body aches, chills and a blocked nose", "I'm an adult with a fever and I'm sneezing",
])
def test_an_adult_with_fever_and_cold_symptoms_gets_fever_products_plus_antihistamine_and_decongestant(text):
    assert _ids(text) == ["col-001", "col-002", "fev-001", "fev-002", "fev-003"]


@pytest.mark.parametrize("text", [
    "I'm an adult with a sore throat", "sneezing and itchy watery eyes, I'm an adult",
])
def test_an_adult_with_cold_symptoms_only_gets_the_cold_products_without_cough_syrups(text):
    assert _ids(text) == ["col-001", "col-002", "col-003"]


def test_cold_and_flu_is_never_offered_next_to_the_paracetamol_fever_tablets():
    for text in ["I have a sore throat and a fever, I'm an adult", "I'm an adult with a fever and I'm sneezing"]:
        assert "col-003" not in _ids(text)


def test_a_child_is_unchanged():
    assert _ids("my 6 year old has a sore throat and fever") == ["col-006", "fev-004"]


def test_a_fever_alone_is_unchanged():
    assert _ids("I'm an adult and I can't smell anything and have a fever") == ["fev-001", "fev-002", "fev-003"]


@pytest.mark.parametrize("text, name", [
    ("I'm an adult with a dry cough", "Cough Suppressant Syrup (Dextromethorphan)"),
    ("I'm an adult with a wet cough", "Guaifenesin Expectorant Syrup"),
])
def test_a_cough_still_reaches_its_own_syrup(text, name):
    # a single recommendation names the product ("... would be a good fit")
    res = client.post("/api/chat", json={"session_id": "s3", "text": text})
    assert res.json()["reply"].startswith(f"{name} would be a good fit")


def test_the_two_step_flow_gives_the_same_filtered_lists():
    first = client.post("/api/chat", json={"session_id": "s2", "text": "I have a sore throat"}).json()["reply"]
    assert "adult" in first
    second = client.post("/api/chat", json={"session_id": "s2", "text": "adult"}).json()["reply"]
    assert sorted(set(re.findall(r"(?:fev|col)-\d{3}", second))) == ["col-001", "col-002", "col-003"]
