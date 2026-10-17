"""Grounding of everyday safety questions (pregnancy, alcohol, driving, ...).

Real bug (Oct 16 live probe): the retriever always returns its nearest chunks
even when they say nothing about the topic asked, and the model filled the gap
from general knowledge while attributing it to "our knowledge base" — including
citing a "Pregnancy" section that does not exist. A safety topic the retrieved
text never mentions now gets a fixed "ask a pharmacist" reply instead.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app import agent, guardrails, safety_topics

pytestmark = pytest.mark.usefixtures("isolated_db")


def _middleware(user_text, turn_state=None):
    return agent._GuardrailMiddleware(
        session_id="s1", user_text=user_text, image_read=False, had_shown_recommendation=False,
        symptom_lookup_grounded=False, turn_state=turn_state if turn_state is not None else {},
    )


def _after_agent(mw, reply_text):
    state = {"messages": [AIMessage(content=reply_text, id="m1")]}
    return mw.after_agent(state, runtime=None)["messages"][0].content


class TestTopicsAsked:
    @pytest.mark.parametrize("question, topic", [
        ("is cetirizine safe during pregnancy", "pregnancy"),
        ("can I take ibuprofen while breastfeeding", "breastfeeding"),
        ("can I drive after taking cetirizine", "driving"),
        ("can I drink alcohol with paracetamol", "alcohol"),
        ("how should I store cough syrup", "storage"),
        ("is ibuprofen suitable for people with asthma", "asthma"),
        ("what happens if I miss a dose of paracetamol", "missed doses"),
        ("how many tablets in a pack of paracetamol", "pack size"),
        ("how many tablets are in a box", "pack size"),
        ("how many ml in the bottle", "pack size"),
        ("how long does paracetamol take to work", "how quickly it works"),
        ("how quickly does ibuprofen start working", "how quickly it works"),
    ])
    def test_each_topic_is_recognised(self, question, topic):
        assert topic in safety_topics.topics_asked(question)

    @pytest.mark.parametrize("question", [
        "I have a fever and a headache", "dosage for paracetamol 500mg", "side effects of cetirizine",
        "order Ibuprofen 200mg Tablets", "my throat hurts and I can't sleep",
        # dosage wording must not be mistaken for pack size or onset
        "how many tablets can I take per day", "how many tablets should I take", "how much paracetamol per dose",
        "how long can I take ibuprofen for", "how long should I wait between doses", "how long does a cold last",
    ])
    def test_ordinary_messages_ask_no_safety_topic(self, question):
        assert safety_topics.topics_asked(question) == []


class TestUncovered:
    def test_a_topic_the_text_never_mentions_is_uncovered(self):
        results = [{"text": "Cetirizine is a second-generation antihistamine."}]
        assert guardrails.uncovered_safety_topics("is cetirizine safe during pregnancy", results) == ["pregnancy"]

    def test_a_topic_the_text_mentions_is_covered(self):
        results = [{"text": "Ask a doctor before use if you are pregnant."}]
        assert guardrails.uncovered_safety_topics("is it safe during pregnancy", results) == []

    def test_only_the_uncovered_topics_are_returned(self):
        results = [{"text": "Avoid alcohol while taking this."}]
        assert guardrails.uncovered_safety_topics("alcohol and pregnancy", results) == ["pregnancy"]

    def test_no_results_means_every_asked_topic_is_uncovered(self):
        assert guardrails.uncovered_safety_topics("pregnancy", []) == ["pregnancy"]

    def test_the_reply_names_the_gap_and_sends_them_to_a_pharmacist(self):
        reply = guardrails.kb_gap_reply(["pregnancy", "alcohol"])
        assert "pregnancy and alcohol" in reply and "pharmacist or doctor" in reply
        assert guardrails.DISCLAIMER in reply


class TestAfterAgentGrounding:
    def test_an_ungrounded_answer_is_replaced(self):
        mw = _middleware(
            "is cetirizine safe during pregnancy",
            {"kb_results": [{"text": "Cetirizine is an antihistamine."}]},
        )
        result = _after_agent(mw, "According to our knowledge base (Source: col-001.md § Pregnancy) it is safe.")
        assert "I don't have information about pregnancy" in result
        assert "safe" not in result.split("pharmacist")[0]

    def test_a_grounded_answer_is_left_alone(self):
        mw = _middleware(
            "can I drink alcohol with paracetamol",
            {"kb_results": [{"text": "Avoid alcohol while taking paracetamol."}]},
        )
        assert _after_agent(mw, "Avoid alcohol.") == "Avoid alcohol."

    def test_an_answer_with_no_lookup_is_checked_against_a_fresh_search(self, monkeypatch):
        monkeypatch.setattr(agent.retrieval, "search", lambda q: [{"text": "Nothing relevant here."}])
        mw = _middleware("how should I store cough syrup")
        assert "storage" in _after_agent(mw, "Keep it in a cool, dry place.")

    def test_a_failed_search_fails_open(self, monkeypatch):
        def _boom(q):
            raise RuntimeError("embeddings down")

        monkeypatch.setattr(agent.retrieval, "search", _boom)
        mw = _middleware("is it safe during pregnancy")
        assert _after_agent(mw, "Ask a doctor.") == "Ask a doctor."

    def test_a_message_with_no_safety_topic_never_triggers_a_search(self, monkeypatch):
        def _boom(q):
            raise AssertionError("no search expected")

        monkeypatch.setattr(agent.retrieval, "search", _boom)
        mw = _middleware("I have a fever")
        assert _after_agent(mw, "Is this for a child or an adult?").startswith("Is this")

    def test_the_lookup_result_is_remembered_for_the_check(self):
        turn_state = {}
        mw = _middleware("anything", turn_state)
        handler = lambda request: SimpleNamespace(content=json.dumps({"results": [{"text": "x"}]}))
        request = SimpleNamespace(tool_call={"name": "lookup_medicine_info", "args": {"query": "q"}, "id": "c1"})
        mw.wrap_tool_call(request, handler)
        assert turn_state["kb_results"] == [{"text": "x"}]


class TestProductFactsTheKnowledgeBaseLacks:
    """The knowledge base has no text on pack size or onset, so any answer
    to those was invented: "usually around 20-30 tablets per pack"."""

    def test_a_dosage_chunk_is_not_mistaken_for_pack_size_coverage(self):
        results = [{"text": "Adults typically take 1-2 tablets every 4-6 hours as needed."}]
        assert guardrails.uncovered_safety_topics("how many tablets in a pack of paracetamol", results) == ["pack size"]

    def test_real_pack_text_would_count_as_covered(self):
        results = [{"text": "Each pack contains 24 tablets."}]
        assert guardrails.uncovered_safety_topics("how many tablets in a pack", results) == []

    def test_real_onset_text_would_count_as_covered(self):
        results = [{"text": "Relief usually begins within 30 minutes."}]
        assert guardrails.uncovered_safety_topics("how long does it take to work", results) == []

    def test_the_real_knowledge_base_has_neither(self):
        from app import retrieval

        for question, label in [("how many tablets in a pack of paracetamol", "pack size"),
                                ("how long does paracetamol take to work", "how quickly it works")]:
            assert guardrails.uncovered_safety_topics(question, retrieval.search(question)) == [label]
