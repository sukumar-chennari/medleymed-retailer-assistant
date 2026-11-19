"""Comparing two products quotes the product information instead of judging.

Real gap (Oct 22-30 probes): "difference between the first two" and "paracetamol
vs ibuprofen which is better" were answered in the model's own words, and the
knowledge base has no comparison text. The reply now puts each product's catalog
entry and its own Overview, Dosage and Warnings text side by side."""

import pytest
from fastapi.testclient import TestClient

from app import agent, comparison, config, data_ingest, guardrails, main, store

CATALOG = store.get_catalog()
client = TestClient(main.app)


def _kb(product_id, heading):
    text = (data_ingest.KB_DIR / f"{product_id}.md").read_text(encoding="utf-8")
    start = text.index(f"## {heading}") + len(f"## {heading}")
    end = text.find("\n## ", start)
    return " ".join(text[start: end if end != -1 else None].split())


class TestIsComparisonRequest:
    @pytest.mark.parametrize("text", [
        "what is the difference between paracetamol and ibuprofen", "Paracetamol 500mg Tablets vs Ibuprofen 200mg Tablets",
        "compare Tylenol and Advil", "which is better, cetirizine or pseudoephedrine", "is paracetamol better than ibuprofen",
        "what's the difference between the first two", "paracetamol or ibuprofen which is safer",
    ])
    def test_comparison_wording(self, text):
        assert comparison.is_comparison_request(text)

    @pytest.mark.parametrize("text", [
        "can I take paracetamol and ibuprofen together", "order ibuprofen instead of paracetamol", "what is ibuprofen",
        "how much is paracetamol vs ibuprofen", "cancel my order", "I have a fever", "",
    ])
    def test_everything_else(self, text):
        assert not comparison.is_comparison_request(text)


class TestTargets:
    def test_ingredients_resolve_to_the_plain_adult_product_with_a_note(self):
        products, notes = comparison.targets("difference between paracetamol and ibuprofen", CATALOG, None)
        assert [p["id"] for p in products] == ["fev-001", "fev-003"]
        assert notes == ["For paracetamol I've used Paracetamol 500mg Tablets."]

    def test_full_names_need_no_note(self):
        products, notes = comparison.targets("Paracetamol 500mg Tablets vs Ibuprofen 200mg Tablets", CATALOG, None)
        assert [p["id"] for p in products] == ["fev-001", "fev-003"] and notes == []

    def test_brands_resolve_to_what_we_carry_and_say_so(self):
        products, notes = comparison.targets("compare Tylenol and Advil", CATALOG, None)
        assert [p["id"] for p in products] == ["fev-001", "fev-003"]
        assert any("We don't carry Tylenol; its active ingredient is paracetamol" in n for n in notes)
        assert any("We don't carry Advil" in n for n in notes)

    def test_a_medicine_we_do_not_carry_and_have_no_equivalent_for_cannot_be_compared(self):
        assert comparison.targets("compare aspirin and ibuprofen", CATALOG, None) is None

    @pytest.mark.parametrize("text, ids", [
        ("what is the difference between the first two", ["fev-001", "fev-002"]),
        ("compare the last two", ["fev-002", "fev-003"]),
        ("difference between the first and the third", ["fev-001", "fev-003"]),
        ("difference between number 1 and number 3", ["fev-001", "fev-003"]),
    ])
    def test_the_list_on_screen(self, text, ids):
        shown = [store.find_product(i) for i in ("fev-001", "fev-002", "fev-003")]
        assert [p["id"] for p in comparison.targets(text, CATALOG, shown)[0]] == ids

    def test_one_product_or_no_list_is_not_a_comparison(self):
        assert comparison.targets("compare ibuprofen", CATALOG, None) is None
        assert comparison.targets("compare the first two", CATALOG, None) is None


class TestTheReply:
    def test_each_products_own_knowledge_base_text_is_quoted_verbatim(self):
        products = [store.find_product("fev-001"), store.find_product("fev-003")]
        reply = comparison.comparison_reply(products, [], guardrails.DISCLAIMER)
        for product_id in ("fev-001", "fev-003"):
            for heading in ("Overview", "Dosage", "Warnings"):
                assert _kb(product_id, heading) in reply, (product_id, heading)
        assert "$4.99" in reply and "$5.29" in reply

    def test_it_refuses_to_pick_a_winner_and_ends_with_the_disclaimer(self):
        reply = comparison.comparison_reply([store.find_product("fev-001"), store.find_product("fev-003")], [], guardrails.DISCLAIMER)
        assert "I can't say one is \"better\"" in reply and reply.rstrip().endswith(guardrails.DISCLAIMER)

    def test_the_gentler_on_the_stomach_line_is_the_knowledge_bases_not_the_models(self):
        # Oct 22 I called this phrase model improvisation; the paracetamol document says it.
        assert "gentler on the stomach than NSAIDs" in _kb("fev-001", "Overview")


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

    def test_a_comparison_is_answered_from_the_documents_without_the_model(self):
        reply = self._chat("s1", "what is the difference between paracetamol and ibuprofen")
        assert _kb("fev-001", "Warnings") in reply and _kb("fev-003", "Warnings") in reply

    def test_the_difference_between_the_first_two_uses_the_list_that_was_shown(self):
        store.set_last_products("s1", [store.find_product(i) for i in ("fev-001", "fev-002", "fev-003")])
        reply = self._chat("s1", "what's the difference between the first two")
        assert "Paracetamol 500mg Tablets (fev-001)" in reply and "Paracetamol Extra Strength 650mg (fev-002)" in reply
        assert "(fev-003)" not in reply

    def test_a_stated_condition_adds_its_caution_to_the_comparison(self):
        store.save_session_messages("s1", [{"role": "user", "content": "I have a stomach ulcer"}])
        reply = self._chat("s1", "paracetamol vs ibuprofen")
        assert "Please be careful with:" in reply and "Ibuprofen 200mg Tablets: you mentioned a stomach ulcer" in reply

    def test_other_messages_are_not_taken_for_comparisons(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert self._chat("s1", "what is ibuprofen used for") == "from the model"
        reply = self._chat("s2", "can I take paracetamol and ibuprofen together")
        assert "how they compare" not in reply and "doesn't say whether" in reply


class TestExactProductsNamedInAComparison:
    """Found Nov 19: "which is stronger paracetamol 650 or ibuprofen" compared the 500mg
    paracetamol ("For paracetamol I've used Paracetamol 500mg Tablets"), "paracetamol 500
    vs 650" was not understood, and "compare ibuprofen and cold and flu" went to the model."""

    @staticmethod
    def _ids(text):
        result = comparison.targets(text, store.get_catalog(), None)
        return None if result is None else [p["id"] for p in result[0]]

    @pytest.mark.parametrize("text, ids", [
        ("which is stronger paracetamol 650 or ibuprofen", ["fev-002", "fev-003"]),
        ("ibuprofen vs paracetamol 650", ["fev-003", "fev-002"]),
        ("paracetamol 500 vs 650", ["fev-001", "fev-002"]),
        ("paracetamol 500mg vs 650mg", ["fev-001", "fev-002"]),
        ("compare ibuprofen and cold and flu", ["fev-003", "col-003"]),
        ("compare the cold & flu and ibuprofen", ["col-003", "fev-003"]),
        ("extra strength paracetamol vs ibuprofen", ["fev-002", "fev-003"]),
        ("compare children's paracetamol syrup and paracetamol 500mg", ["fev-004", "fev-001"]),
    ])
    def test_the_named_products_are_the_ones_compared(self, text, ids):
        assert self._ids(text) == ids

    def test_no_default_product_note_when_the_strength_was_named(self):
        _, notes = comparison.targets("which is stronger paracetamol 650 or ibuprofen", store.get_catalog(), None)
        assert not any(n.startswith("For paracetamol") for n in notes)

    @pytest.mark.parametrize("text, ids", [
        ("paracetamol vs ibuprofen", ["fev-001", "fev-003"]),
        ("difference between cetirizine and pseudoephedrine", ["col-001", "col-002"]),
    ])
    def test_unqualified_ingredients_keep_using_the_default_product(self, text, ids):
        assert self._ids(text) == ids
        _, notes = comparison.targets(text, store.get_catalog(), None)
        assert (text == "paracetamol vs ibuprofen") == any("I've used" in n for n in notes)


def test_the_route_compares_the_650_when_the_650_was_named(monkeypatch):
    # end to end through /api/chat (the model must not be reached)
    def boom(*args, **kwargs):
        raise AssertionError("the model must not be reached")
    monkeypatch.setattr(main, "run_turn", boom)
    reply = TestClient(main.app).post("/api/chat", json={"session_id": "cmp", "text": "which is stronger paracetamol 650 or ibuprofen"}).json()["reply"]
    assert "Paracetamol Extra Strength 650mg (fev-002)" in reply and "Ibuprofen 200mg Tablets (fev-003)" in reply
    assert "(fev-001)" not in reply
