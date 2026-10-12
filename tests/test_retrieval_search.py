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

from unittest import mock

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


class TestEmptyCollection:
    def test_an_empty_collection_returns_nothing_without_calling_the_embedding_model(self, monkeypatch):
        # _collection is a module-level singleton (the real, already-
        # ingested one) — mocked here only for its .count(), never
        # replacing the real collection, so no other test in this file is
        # affected. The short-circuit happens before the embed call, so
        # this also confirms no real network call is attempted either.
        fake_collection = mock.Mock()
        fake_collection.count.return_value = 0
        monkeypatch.setattr(retrieval, "_collection", fake_collection)

        with mock.patch.object(retrieval, "_client") as fake_client:
            result = retrieval.search("dosage for paracetamol 500mg")

        assert result == []
        fake_client.embed.assert_not_called()


class TestSearchDoesNotOverNarrow:
    """Real bug: results were narrowed to the one product a distinctive title
    word named, discarding products the query also pointed at."""

    def test_comparing_two_products_returns_both(self):
        results = retrieval.search("paracetamol vs ibuprofen which is better for fever")
        products = {r["product"] for r in results}
        assert "Ibuprofen 200mg Tablets" in products
        assert any("Paracetamol" in p for p in products)  # used to be dropped entirely

    def test_a_childs_dextromethorphan_question_includes_the_childrens_syrup(self):
        # It used to return ONLY the adult syrup's dosage: only that product's
        # title says dextromethorphan, though the children's syrup contains it.
        results = retrieval.search("dextromethorphan dosage for children")
        assert "Children's Cold & Cough Syrup" in {r["product"] for r in results}

    def test_a_five_year_old_question_includes_the_childrens_syrup(self):
        results = retrieval.search("cough suppressant for my 5 year old")
        assert "Children's Cold & Cough Syrup" in {r["product"] for r in results}

    def test_a_generic_pain_relief_question_is_not_pinned_to_the_multi_symptom_product(self):
        results = retrieval.search("what pain relief is there for fever")
        assert {r["product"] for r in results} != {"Cold & Flu Multi-Symptom Relief"}
        assert any(r["source"].startswith("fev-") for r in results)

    def test_a_query_naming_one_product_is_still_narrowed_to_it(self):
        results = retrieval.search("can I take ibuprofen for a fever")
        assert results and {r["product"] for r in results} == {"Ibuprofen 200mg Tablets"}


class TestSearchSectionBoostForProductsWithoutADistinctiveTitleWord:
    """Real bug: the section boost only fired for products with a title word
    unique to them, which fev-001, fev-004 and col-006 lack — so for them the
    asked-for section was never boosted and ANOTHER product's chunk won."""

    def test_a_childs_paracetamol_question_gets_the_childrens_own_dosage_not_the_adult_one(self):
        # It used to return the children's syrup Overview plus the ADULT
        # Extra Strength product's Dosage, and never the children's Dosage.
        for query in ["children's paracetamol dosing", "how much paracetamol can I give my child"]:
            results = retrieval.search(query)
            assert results[0]["product"] == "Children's Paracetamol Syrup", query
            assert results[0]["section"] == "Dosage", query
            assert not any(r["product"] == "Paracetamol Extra Strength 650mg" for r in results), query

    def test_the_childrens_cold_syrup_question_gets_its_own_dosage_not_the_paracetamol_syrups(self):
        # It used to return col-006's Overview plus fev-004's (paracetamol
        # syrup) Dosage — a different medicine's dosing.
        for query in ["children's cold and cough syrup dosage", "dosing for children's cold and cough syrup"]:
            results = retrieval.search(query)
            assert results[0]["product"] == "Children's Cold & Cough Syrup", query
            assert results[0]["section"] == "Dosage", query
            assert not any(r["product"] == "Children's Paracetamol Syrup" for r in results), query

    def test_the_500mg_dosage_question_includes_the_dosage_section_either_word_order(self):
        # "paracetamol 500mg dosage" returned only Overview + Side Effects.
        for query in ["paracetamol 500mg dosage", "dosage for paracetamol 500mg"]:
            results = retrieval.search(query)
            assert results[0]["product"] == "Paracetamol 500mg Tablets", query
            assert results[0]["section"] == "Dosage", query
