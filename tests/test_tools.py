"""Regression tests for the pure, read-only parts of app/tools.py.

Scoped to functions with no database side effects (classify_categories,
classify, lookup_symptom, _clamp_quantity) — lookup_symptom only reads the
catalog (loaded once from catalog.json), never writes. Deliberately leaves
out save_address/create_order/cancel_order/check_order_status here: those
write to the real app.db, and this demo has one shared "demo_user" with no
test-database isolation set up, so exercising them here would either
pollute real order history used for the live demo or need DB-path
monkeypatching disproportionate to this project's scale.
"""

import json

import pytest

from app import store, tools


class TestClassifyCategories:
    def test_fever_only(self):
        assert tools.classify_categories("I have a fever") == ["fever"]

    def test_cold_only(self):
        assert tools.classify_categories("I have a runny nose") == ["cold"]

    def test_exact_match_shortcuts(self):
        assert tools.classify_categories("fever") == ["fever"]
        assert tools.classify_categories("cold") == ["cold"]
        assert tools.classify_categories("  Cold  ") == ["cold"]

    def test_both_categories_in_one_message(self):
        # Both must come back, not just whichever matched first — see
        # _category_hit's own docstring for the bug this covers.
        categories = tools.classify_categories("nose block and fever")
        assert "fever" in categories
        assert "cold" in categories

    def test_no_match(self):
        assert tools.classify_categories("fix my code") == []
        assert tools.classify_categories("what's the weather today") == []

    def test_typo_tolerance_for_long_words(self):
        assert tools.classify_categories("sneazing all day") == ["cold"]
        assert tools.classify_categories("bad cetrizine reaction") == ["cold"]

    def test_short_word_fuzzy_matching_regression(self):
        # "code" vs "cold" scores higher under naive fuzzy matching than the
        # real typo "runnig" vs "runny" does — this must NOT classify as cold.
        assert tools.classify_categories("fix my code") == []

    def test_unrelated_word_near_congestion_regression(self):
        # A phone-line "reconnection" letter previously fuzzy-matched
        # "congestion" at the old 0.8 cutoff — a real false positive found
        # via a photographed unrelated document, not a hypothetical.
        assert tools.classify_categories("your line reconnection is confirmed") == []

    def test_short_keyword_inside_an_unrelated_word_does_not_count(self):
        # Real bug: short single-word keywords ("flu", "nasal") used to
        # match as raw substrings of ANY word containing them — "the
        # fluent speaker" wrongly classified as a cold symptom mention via
        # "flu" inside "fluent", with zero real symptom content.
        assert tools.classify_categories("the fluent speaker") == []
        assert tools.classify_categories("I feel confluent today") == []
        assert tools.classify_categories("the flute solo was great") == []
        assert tools.classify_categories("nasally speaking, that was odd") == []

    def test_short_keyword_as_a_real_standalone_word_still_counts(self):
        assert tools.classify_categories("I have the flu") == ["cold"]
        assert tools.classify_categories("my nose feels stuffy and nasal") == ["cold"]


class TestClassify:
    def test_returns_first_category(self):
        assert tools.classify("I have a fever") == "fever"
        assert tools.classify("I have a cough") == "cold"

    def test_returns_none_for_no_match(self):
        assert tools.classify("fix my code") is None


class TestLookupSymptom:
    def test_matched_fever_symptom_returns_real_products(self):
        result = json.loads(tools.lookup_symptom("fever"))
        assert result["matched"] is True
        assert result["category"] == "fever"
        assert len(result["products"]) > 0
        assert all("id" in p and "name" in p for p in result["products"])

    def test_unmatched_symptom_returns_no_products(self):
        result = json.loads(tools.lookup_symptom("fix my code"))
        assert result["matched"] is False
        assert "products" not in result


class TestLookupMedicineInfo:
    """This is a deterministic wrapper around retrieval.search (itself
    already covered end-to-end in test_retrieval_search.py) — these tests
    just confirm tools.lookup_medicine_info shapes that result correctly
    at the tool-call boundary, using the real knowledge base."""

    def test_matched_query_returns_real_results(self):
        result = json.loads(tools.lookup_medicine_info("dosage for paracetamol 500mg"))
        assert result["results"]
        assert result["results"][0]["product"] == "Paracetamol 500mg Tablets"

    def test_unmatched_query_returns_an_empty_result_with_a_message(self):
        result = json.loads(tools.lookup_medicine_info("credit card refund policy"))
        assert result == {
            "results": [],
            "message": "No information found in our knowledge base for that.",
        }


class TestClampQuantity:
    def test_normal_quantity_passes_through(self):
        assert tools._clamp_quantity(1) == (1, None)

    def test_zero_or_negative_clamps_to_one_with_a_reason(self):
        # Real bug fix: this used to clamp silently (was_clamped=False),
        # asymmetric with the over-the-limit case below, which always
        # surfaced a note to the user.
        quantity, reason = tools._clamp_quantity(0)
        assert quantity == 1
        assert reason is not None
        assert "limit" not in reason  # not a limit — the input was just invalid

        quantity, reason = tools._clamp_quantity(-5)
        assert quantity == 1
        assert reason is not None

    def test_over_the_limit_clamps_and_flags(self):
        quantity, reason = tools._clamp_quantity(tools.MAX_QUANTITY_PER_ORDER + 5)
        assert quantity == tools.MAX_QUANTITY_PER_ORDER
        assert reason is not None
        assert "limit" in reason


class TestSaveAddressRefusesAnEmail:
    """Real bug (fuzz of the real chat model, Nov 6): the model called
    save_address("x@y.com") and the email became the shipping address."""

    @pytest.mark.parametrize("value", ["x@y.com", "  sukumar123@example.com ", "first.last+tag@sub.example.co.uk"])
    def test_an_email_is_not_saved_as_an_address(self, isolated_db, value):
        result = json.loads(tools.save_address(value))
        assert result["saved"] is False and "email address" in result["error"]
        assert store.get_address("demo_user") in (None, "")

    @pytest.mark.parametrize("value", ["5 Mid Rd, Pune, a@b.com", "10 High St, Leeds", "hyderabad"])
    def test_an_address_that_merely_mentions_one_or_has_none_is_saved(self, isolated_db, value):
        assert json.loads(tools.save_address(value))["saved"] is True


class TestSaveAddressRefusesAToolName:
    """Real bug (fuzz of the real chat model, Nov 14): with an order waiting for its
    address, "both of them" made the model call save_address("get_saved_address") —
    one of its own tool names became the shipping address."""

    @pytest.mark.parametrize("value", [
        "get_saved_address", "check_order_status", "  cancel_order ", "get_saved_address()", "start_order(product_id='fev-001')",
        "reorder_last", "Lookup_Symptom",
    ])
    def test_a_tool_name_is_not_saved_as_an_address(self, isolated_db, value):
        result = json.loads(tools.save_address(value))
        assert result["saved"] is False and "not a shipping address" in result["error"]
        assert store.get_address("demo_user") in (None, "")

    @pytest.mark.parametrize("value", [
        "10 High St, Leeds", "hyderabad", "Flat 4_B, 22 MG Road", "Kolkata", "12 Baker Street, London (near the park)",
    ])
    def test_real_addresses_are_still_saved(self, isolated_db, value):
        assert json.loads(tools.save_address(value))["saved"] is True
