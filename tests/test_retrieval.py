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

import json
from unittest import mock

from app import data_ingest, retrieval


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


class TestLoadCollection:
    """_load_collection is the self-invalidating cache in front of the RAG
    index — the actual correctness question here is "does it rebuild
    exactly when it should," not "does build_index() work" (that needs a
    live embedding call and is already exercised by CI's cold-start path,
    see .github/workflows/tests.yml). So build_index and chromadb's client
    are both mocked out here entirely: no real embedding call, no real
    Chroma write, just verifying which branch each cache state takes.
    Without this, a knowledge-base edit that silently failed to invalidate
    the cache would serve stale embeddings indefinitely with nothing
    catching it."""

    def _mock_chroma_client(self, mock_client_cls, get_collection_result):
        mock_client_cls.return_value.get_collection = mock.Mock(side_effect=get_collection_result) if isinstance(
            get_collection_result, list
        ) else mock.Mock(return_value=get_collection_result)

    def test_no_cached_metadata_triggers_a_rebuild(self, monkeypatch, tmp_path):
        monkeypatch.setattr(data_ingest, "META_PATH", tmp_path / "kb_meta.json")
        fake_collection = mock.Mock(name="fake_collection")
        with mock.patch.object(data_ingest, "build_index") as mock_build, \
                mock.patch("app.retrieval.chromadb.PersistentClient") as mock_client_cls:
            self._mock_chroma_client(mock_client_cls, fake_collection)
            result = retrieval._load_collection()
        mock_build.assert_called_once()
        assert result is fake_collection

    def test_stale_content_hash_triggers_a_rebuild(self, monkeypatch, tmp_path):
        meta_path = tmp_path / "kb_meta.json"
        meta_path.write_text(json.dumps({"content_hash": "a-stale-hash-from-before-a-kb-edit"}))
        monkeypatch.setattr(data_ingest, "META_PATH", meta_path)
        fake_collection = mock.Mock(name="fake_collection")
        with mock.patch.object(data_ingest, "build_index") as mock_build, \
                mock.patch("app.retrieval.chromadb.PersistentClient") as mock_client_cls:
            self._mock_chroma_client(mock_client_cls, fake_collection)
            result = retrieval._load_collection()
        mock_build.assert_called_once()
        assert result is fake_collection

    def test_matching_hash_but_missing_collection_triggers_a_rebuild(self, monkeypatch, tmp_path):
        # Metadata says the cache should be valid, but the actual Chroma
        # collection is gone (e.g. chroma_db/ deleted without also
        # deleting kb_meta.json) — must still rebuild, not raise.
        real_hash = data_ingest.compute_content_hash()
        meta_path = tmp_path / "kb_meta.json"
        meta_path.write_text(json.dumps({"content_hash": real_hash}))
        monkeypatch.setattr(data_ingest, "META_PATH", meta_path)
        fake_collection = mock.Mock(name="fake_collection")
        with mock.patch.object(data_ingest, "build_index") as mock_build, \
                mock.patch("app.retrieval.chromadb.PersistentClient") as mock_client_cls:
            self._mock_chroma_client(mock_client_cls, [Exception("no such collection"), fake_collection])
            result = retrieval._load_collection()
        mock_build.assert_called_once()
        assert result is fake_collection

    def test_matching_hash_and_present_collection_skips_the_rebuild(self, monkeypatch, tmp_path):
        real_hash = data_ingest.compute_content_hash()
        meta_path = tmp_path / "kb_meta.json"
        meta_path.write_text(json.dumps({"content_hash": real_hash}))
        monkeypatch.setattr(data_ingest, "META_PATH", meta_path)
        fake_collection = mock.Mock(name="fake_collection")
        with mock.patch.object(data_ingest, "build_index") as mock_build, \
                mock.patch("app.retrieval.chromadb.PersistentClient") as mock_client_cls:
            self._mock_chroma_client(mock_client_cls, fake_collection)
            result = retrieval._load_collection()
        mock_build.assert_not_called()
        assert result is fake_collection
