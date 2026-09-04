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

from app import tools


class TestClassifyCategories:
    def test_fever_only(self):
        assert tools.classify_categories("I have a fever") == ["fever"]

    def test_cold_only(self):
        assert tools.classify_categories("I have a runny nose") == ["cold"]

    def test_both_categories_in_one_message(self):
        # Both must come back, not just whichever matched first — see
        # _category_hit's own docstring for the bug this covers.
        categories = tools.classify_categories("nose block and feverih")
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


class TestClampQuantity:
    def test_normal_quantity_passes_through(self):
        assert tools._clamp_quantity(1) == (1, False)

    def test_zero_or_negative_clamps_to_one(self):
        assert tools._clamp_quantity(0) == (1, False)
        assert tools._clamp_quantity(-5) == (1, False)

    def test_over_the_limit_clamps_and_flags(self):
        quantity, clamped = tools._clamp_quantity(tools.MAX_QUANTITY_PER_ORDER + 5)
        assert quantity == tools.MAX_QUANTITY_PER_ORDER
        assert clamped is True
