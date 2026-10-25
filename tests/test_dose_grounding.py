"""Dose figures must come from the knowledge base (guardrails.ungrounded_dose_figures).

Real bug (Oct 25 probe): asked for the maximum daily dose of ibuprofen the model
gave the right 6 tablets / 1200 mg but added "3-4 tablets (600-800 mg) every 4-6
hours" — the knowledge base says 1-2 tablets (200-400 mg) per dose, so the
invented single dose was double the documented one."""

import pytest
from langchain_core.messages import AIMessage

from app import agent, guardrails, retrieval

pytestmark = pytest.mark.usefixtures("isolated_db")

IBUPROFEN_DOSAGE = {
    "source": "fev-003.md", "section": "Dosage", "product": "Ibuprofen 200mg Tablets",
    "text": "## Dosage\nAdults typically take 1-2 tablets (200-400mg) every 4-6 hours as needed, not\n"
            "exceeding 6 tablets (1200mg) in 24 hours.",
}
BAD = "The maximum is 6 tablets (1200mg) in 24 hours, and you can take 3-4 tablets (600-800mg) every 4-6 hours."
GOOD = "Adults take 1-2 tablets (200-400mg) every 4-6 hours, not more than 6 tablets (1200mg) in 24 hours."


class TestUngroundedDoseFigures:
    def test_an_invented_dose_is_flagged(self):
        assert guardrails.ungrounded_dose_figures(BAD, [IBUPROFEN_DOSAGE]) == ["3-4 tablets", "600-800mg"]

    def test_a_dose_the_knowledge_base_states_passes(self):
        assert guardrails.ungrounded_dose_figures(GOOD, [IBUPROFEN_DOSAGE]) == []

    def test_a_figure_stated_only_elsewhere_in_the_product_document_passes(self):
        # Retrieval returned only the Warnings chunk; the 4-6 hours is in the
        # same product's Dosage section.
        warnings_only = {"source": "fev-001.md", "section": "Warnings", "text": "## Warnings\nDo not combine."}
        assert guardrails.ungrounded_dose_figures("wait 4-6 hours", [warnings_only]) == []
        assert guardrails.ungrounded_dose_figures("wait 9 hours", [warnings_only]) == ["9 hours"]

    def test_a_number_the_customer_gave_is_not_an_invented_dose(self):
        reply = "You took 2 tablets 1 hour ago."
        assert guardrails.ungrounded_dose_figures(reply, [IBUPROFEN_DOSAGE], "I took 2 tablets 1 hour ago") == []

    def test_the_strengths_in_our_product_names_are_not_dose_claims(self):
        reply = "I recommend Paracetamol Extra Strength 650mg or Cetirizine 10mg Antihistamine Tablets."
        assert guardrails.ungrounded_dose_figures(reply, []) == []

    def test_text_without_figures_is_never_flagged(self):
        assert guardrails.ungrounded_dose_figures("Follow the label and ask a pharmacist.", [IBUPROFEN_DOSAGE]) == []

    def test_a_citation_is_not_mistaken_for_a_figure(self):
        assert guardrails.ungrounded_dose_figures("(Source: fev-003.md § Dosage)", [IBUPROFEN_DOSAGE]) == []


class TestKbQuoteReply:
    def test_it_quotes_the_dosage_chunk_with_its_source(self):
        reply = guardrails.kb_quote_reply([IBUPROFEN_DOSAGE])
        assert "1-2 tablets (200-400mg) every 4-6 hours" in reply and "## Dosage" not in reply
        assert "(Source: fev-003.md § Dosage)" in reply and guardrails.DISCLAIMER in reply
        assert "\n" not in reply.split("Ibuprofen 200mg Tablets:")[1].split("\n\n")[0]  # wrapped lines are joined

    def test_without_a_dosage_chunk_it_quotes_the_products_dosage_section(self):
        warnings_only = {"source": "fev-001.md", "section": "Warnings", "product": "Paracetamol 500mg Tablets", "text": "## Warnings\nx"}
        reply = guardrails.kb_quote_reply([warnings_only])
        assert "every 4-6 hours" in reply and "(Source: fev-001.md § Dosage)" in reply


def _after_agent(turn_state, reply_text, user_text="what is the maximum dose of ibuprofen per day"):
    mw = agent._GuardrailMiddleware(
        session_id="s1", user_text=user_text, image_read=False, had_shown_recommendation=False,
        symptom_lookup_grounded=False, turn_state=turn_state,
    )
    state = {"messages": [AIMessage(content=reply_text, id="m1")]}
    return mw.after_agent(state, runtime=None)["messages"][0].content


class TestAfterAgent:
    def test_an_invented_dose_is_replaced_by_the_knowledge_base_text(self):
        result = _after_agent({"kb_results": [IBUPROFEN_DOSAGE]}, BAD)
        assert "3-4 tablets" not in result and "600-800" not in result
        assert "1-2 tablets (200-400mg) every 4-6 hours" in result

    def test_a_grounded_dose_is_left_alone(self):
        assert _after_agent({"kb_results": [IBUPROFEN_DOSAGE]}, GOOD).startswith(GOOD)

    def test_a_reply_with_no_figures_triggers_no_search(self, monkeypatch):
        def _boom(q):
            raise AssertionError("no search expected")

        monkeypatch.setattr(agent.retrieval, "search", _boom)
        assert _after_agent({}, "Please ask a pharmacist.", "hello").startswith("Please ask a pharmacist.")

    def test_a_dose_with_no_lookup_is_checked_against_a_fresh_search(self, monkeypatch):
        monkeypatch.setattr(agent.retrieval, "search", lambda q: [IBUPROFEN_DOSAGE])
        result = _after_agent({}, "Take 3-4 tablets every 4-6 hours.", "how much ibuprofen")
        assert "3-4 tablets" not in result and "1-2 tablets" in result

    def test_when_there_is_nothing_to_quote_it_fails_open(self, monkeypatch):
        monkeypatch.setattr(agent.retrieval, "search", lambda q: [])
        assert "3-4 tablets" in _after_agent({}, "Take 3-4 tablets.", "how much ibuprofen")

    def test_a_failed_search_fails_open(self, monkeypatch):
        def _boom(q):
            raise RuntimeError("embeddings down")

        monkeypatch.setattr(agent.retrieval, "search", _boom)
        assert "3-4 tablets" in _after_agent({}, "Take 3-4 tablets.", "how much ibuprofen")

    def test_the_real_retrieval_for_the_probe_question_flags_the_invented_dose(self):
        question = "what is the maximum dose of ibuprofen per day"
        results = retrieval.search(question)
        assert guardrails.ungrounded_dose_figures(BAD, results, question) == ["3-4 tablets", "600-800mg"]
        assert guardrails.ungrounded_dose_figures(GOOD, results, question) == []
