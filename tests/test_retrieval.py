"""Regression tests for the pure per-product matching logic in
app/retrieval.py — the hybrid keyword+semantic layer that's had two real,
previously-shipped bugs: SECTION_BOOST firing on words shared across
multiple products once the catalog grew (col-004/005/006 all being
cough/cold syrups), and _identify_single_product not existing at all,
which let "can I take ibuprofen for a fever" retrieve and cite both
Ibuprofen and Paracetamol chunks together.

Importing app.retrieval computes _distinctive_words once from the real,
already-ingested Chroma collection (see retrieval.py's module-level
`_distinctive_words = _distinctive_words_by_title(_collection)`) — these
tests exercise that real, current catalog data rather than a mock, so a
future catalog change that reintroduces a title collision (e.g. two
products sharing a "distinctive" word) fails here instead of only
surfacing live in a chat reply.
"""

from app import retrieval


class FakeCollection:
    """Minimal stand-in for a Chroma collection — _distinctive_words_by_title
    only ever calls .get()["metadatas"], so nothing else needs mocking."""

    def __init__(self, titles):
        self._titles = titles

    def get(self):
        return {"metadatas": [{"title": t} for t in self._titles]}


class TestMgStrengths:
    def test_extracts_a_single_strength(self):
        assert retrieval._mg_strengths("Paracetamol 500mg Tablets") == {"500mg"}

    def test_normalizes_a_space_before_mg(self):
        assert retrieval._mg_strengths("Paracetamol 500 mg Tablets") == {"500mg"}

    def test_extracts_multiple_strengths(self):
        assert retrieval._mg_strengths("500mg or 650mg") == {"500mg", "650mg"}

    def test_no_strength_present(self):
        assert retrieval._mg_strengths("Guaifenesin Expectorant Syrup") == set()


class TestDistinctiveWordsByTitle:
    def test_word_shared_across_titles_is_excluded(self):
        # "syrup" appears in both titles, so it's not a safe per-product
        # anchor for either one.
        words = retrieval._distinctive_words_by_title(
            FakeCollection(["Cough Suppressant Syrup", "Guaifenesin Expectorant Syrup"])
        )
        assert "syrup" not in words["Cough Suppressant Syrup"]
        assert "syrup" not in words["Guaifenesin Expectorant Syrup"]

    def test_word_unique_to_one_title_is_included(self):
        words = retrieval._distinctive_words_by_title(
            FakeCollection(["Cough Suppressant Syrup", "Guaifenesin Expectorant Syrup"])
        )
        assert "suppressant" in words["Cough Suppressant Syrup"]
        assert "guaifenesin" in words["Guaifenesin Expectorant Syrup"]

    def test_short_words_are_excluded(self):
        # Only 5+ letter words are considered safe anchors at all.
        words = retrieval._distinctive_words_by_title(FakeCollection(["Cold & Flu Relief"]))
        assert "cold" not in words["Cold & Flu Relief"]
        assert "flu" not in words["Cold & Flu Relief"]
        assert "relief" in words["Cold & Flu Relief"]


class TestTitleMentioned:
    def test_distinctive_word_matches_its_own_product(self):
        assert retrieval._title_mentioned("ibuprofen 200mg tablets", "Ibuprofen 200mg Tablets")

    def test_shared_word_does_not_match_on_its_own(self):
        # "cough" and "syrup" are shared across col-004/005/006's titles in
        # the real catalog, so neither alone should register as "mentioning"
        # any specific one of them.
        assert not retrieval._title_mentioned("cough syrup", "Guaifenesin Expectorant Syrup")


class TestIdentifySingleProduct:
    def test_identifies_the_named_product(self):
        # The exact real regression this function exists to fix: retrieval
        # used to return both Ibuprofen and Paracetamol chunks for this
        # query, and the model cited both.
        assert retrieval._identify_single_product("can i take ibuprofen for a fever") == "Ibuprofen 200mg Tablets"

    def test_identifies_a_multi_word_distinctive_match(self):
        result = retrieval._identify_single_product("dosage for cough suppressant syrup")
        assert result == "Cough Suppressant Syrup (Dextromethorphan)"

    def test_ambiguous_shared_words_resolve_to_none(self):
        # "cough" and "syrup" alone are shared across three real catalog
        # products (col-004/005/006) — this must stay unresolved rather than
        # guessing one, exactly the SECTION_BOOST cross-catalog bug this
        # project hit once the catalog grew.
        assert retrieval._identify_single_product("dosage for cough syrup") is None

    def test_generic_query_resolves_to_none(self):
        assert retrieval._identify_single_product("what's the weather today") is None
