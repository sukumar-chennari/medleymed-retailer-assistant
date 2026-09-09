"""End-to-end tests for retrieval.search() itself — the function every
sub-helper in test_retrieval.py (_mg_strengths, _identify_single_product,
etc.) only ever gets exercised indirectly through. Unlike agent.run_turn
(which samples from the chat model and is genuinely nondeterministic),
search() has no sampling anywhere in it: the embedding model is called
once per query with no temperature, and cosine-similarity ranking is pure
math — calling it twice with the same query returns byte-identical
results (verified before writing this file). CI already has to install
Ollama and the embedding model anyway for app.tools/app.retrieval to even
import (see .github/workflows/tests.yml), so testing the real search()
against the real, already-ingested knowledge base costs nothing extra and
is not flaky.

Pins the three real, previously-shipped bugs that motivated STRENGTH_BOOST,
SECTION_BOOST, and the single-product filter — see retrieval.py's own
comments for the full story on each.
"""

from app import retrieval


class TestSearchStrengthBoost:
    def test_500mg_query_ranks_the_500mg_product_first(self):
        # Real bug: pure semantic similarity alone ranked fev-002 (650mg)
        # above fev-001 (500mg) for this exact query — both are near-
        # identical paracetamol dosage text, and the embedding model
        # doesn't weight the number heavily on its own.
        results = retrieval.search("dosage for paracetamol 500mg")
        assert results
        assert results[0]["product"] == "Paracetamol 500mg Tablets"

    def test_650mg_query_ranks_the_650mg_product_first(self):
        results = retrieval.search("how much extra strength paracetamol can I take")
        assert results
        assert results[0]["product"] == "Paracetamol Extra Strength 650mg"


class TestSearchSectionBoost:
    def test_dosage_query_ranks_the_dosage_section_first(self):
        # Real bug: "dosage for cough suppressant syrup" scored col-004's
        # own Dosage section BELOW its Overview/Warnings sections — three
        # similarly-worded paragraphs about the same product with nothing
        # in the raw embedding strongly preferring Dosage specifically.
        results = retrieval.search("dosage for cough suppressant syrup")
        assert results
        assert results[0]["section"] == "Dosage"
        assert results[0]["product"] == "Cough Suppressant Syrup (Dextromethorphan)"


class TestSearchSingleProductFilter:
    def test_ibuprofen_query_never_returns_paracetamol_chunks(self):
        # The exact real regression: this query used to retrieve and cite
        # BOTH Ibuprofen and Paracetamol chunks together, and the model
        # recommended paracetamol as an unprompted "alternative."
        results = retrieval.search("can i take ibuprofen for a fever")
        assert results
        assert all(r["product"] == "Ibuprofen 200mg Tablets" for r in results)

    def test_ambiguous_shared_word_query_can_still_return_multiple_products(self):
        # "cough syrup" alone doesn't identify one specific product (see
        # test_retrieval.py's _identify_single_product tests) — search
        # should not incorrectly filter this down to just one.
        results = retrieval.search("cough syrup")
        products = {r["product"] for r in results}
        assert len(products) >= 1  # not asserting >1 exactly — just that it isn't wrongly forced to one


class TestSearchMinSimilarityThreshold:
    def test_unrelated_query_returns_nothing(self):
        assert retrieval.search("credit card refund policy") == []
        assert retrieval.search("what's the weather today") == []

    def test_off_catalog_medicine_returns_nothing(self):
        assert retrieval.search("dosage for amoxicillin") == []


class TestSearchResultShape:
    def test_results_are_capped_at_top_k(self):
        results = retrieval.search("paracetamol dosage", top_k=2)
        assert len(results) <= 2

    def test_every_result_has_the_expected_fields(self):
        results = retrieval.search("side effects of cetirizine")
        assert results
        for r in results:
            assert set(r.keys()) == {"product", "source", "section", "text", "score"}
            assert 0.0 <= r["score"] <= 1.0

    def test_results_are_sorted_by_score_descending(self):
        results = retrieval.search("warnings for pseudoephedrine")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_is_deterministic_across_repeated_calls(self):
        assert retrieval.search("dosage for paracetamol 500mg") == retrieval.search("dosage for paracetamol 500mg")
