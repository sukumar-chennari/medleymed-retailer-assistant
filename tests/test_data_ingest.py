"""Regression tests for app/data_ingest.py: document loading, the
header-based chunking strategy, and build_index itself — the only real
network dependency there is the embedding call (_client.embed), mocked
below the same way smtplib.SMTP and the Gemini client were mocked
elsewhere, so no live Ollama call is needed and no real chroma_db/
kb_meta.json is touched (CHROMA_DIR/META_PATH redirected to a temp
directory). This still exercises the real chunking, real Chroma writes,
and the real content-hash metadata — just not the embedding model itself,
which CI's cold-start path (see .github/workflows/tests.yml) already
exercises for real when it needs to.
"""

import json
from types import SimpleNamespace

import chromadb

from app import data_ingest


class TestLoadDocuments:
    def test_returns_every_real_knowledge_base_file(self):
        docs = data_ingest.load_documents()
        filenames = {name for name, _ in docs}
        assert "fev-001.md" in filenames
        assert "col-001.md" in filenames
        assert len(docs) == 10  # one per catalog product

    def test_sorted_for_deterministic_order(self):
        docs = data_ingest.load_documents()
        filenames = [name for name, _ in docs]
        assert filenames == sorted(filenames)

    def test_each_file_has_real_text_content(self):
        docs = data_ingest.load_documents()
        assert all(text.strip() for _, text in docs)


class TestComputeContentHash:
    def test_deterministic_across_calls(self):
        assert data_ingest.compute_content_hash() == data_ingest.compute_content_hash()

    def test_is_a_sha256_hex_digest(self):
        digest = data_ingest.compute_content_hash()
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


SAMPLE_DOC = """\
# Sample Medicine 100mg Tablets

## Overview
General overview text.

## Dosage
Take one tablet every 6 hours.

## Common Side Effects
Mild drowsiness.

## Warnings
Do not exceed the labeled dose.
"""


class TestChunkDocument:
    def test_splits_into_one_chunk_per_section(self):
        chunks = data_ingest.chunk_document("sample.md", SAMPLE_DOC)
        sections = [c["section"] for c in chunks]
        assert sections == ["Overview", "Dosage", "Common Side Effects", "Warnings"]

    def test_every_chunk_carries_source_and_doc_title(self):
        chunks = data_ingest.chunk_document("sample.md", SAMPLE_DOC)
        assert all(c["source"] == "sample.md" for c in chunks)
        assert all(c["title"] == "Sample Medicine 100mg Tablets" for c in chunks)

    def test_chunk_text_includes_its_own_header(self):
        chunks = data_ingest.chunk_document("sample.md", SAMPLE_DOC)
        dosage_chunk = next(c for c in chunks if c["section"] == "Dosage")
        assert dosage_chunk["text"].startswith("## Dosage")
        assert "every 6 hours" in dosage_chunk["text"]

    def test_falls_back_to_filename_when_no_title_line(self):
        text = "## Overview\nNo title line above this.\n"
        chunks = data_ingest.chunk_document("untitled.md", text)
        assert chunks[0]["title"] == "untitled.md"

    def test_ignores_content_before_the_first_section_header(self):
        # The title line itself (a "# " line, not "## ") must not become a
        # bogus extra chunk.
        chunks = data_ingest.chunk_document("sample.md", SAMPLE_DOC)
        assert len(chunks) == 4
        assert not any(c["text"].startswith("# Sample Medicine") for c in chunks)

    def test_real_knowledge_base_files_all_chunk_into_the_standard_four_sections(self):
        for filename, text in data_ingest.load_documents():
            chunks = data_ingest.chunk_document(filename, text)
            sections = {c["section"] for c in chunks}
            assert sections == {"Overview", "Dosage", "Common Side Effects", "Warnings"}, filename


class TestChunkDocumentIsLossless:
    """Real bug: chunk_document silently DROPPED anything that didn't start
    with exactly "## " — a section written "##Dosage" vanished from the index,
    text before the first heading was discarded, and a file with no "##"
    headings produced zero chunks, all without an error. One typo in a
    knowledge-base edit would silently delete a medicine's dosage from the
    assistant. (The 10 real files are fine; this guards future edits.)"""

    def test_a_heading_without_a_space_is_kept(self):
        chunks = data_ingest.chunk_document("x.md", "# T\n##Overview\nO text.\n## Dosage\nD text.\n")
        assert [c["section"] for c in chunks] == ["Overview", "Dosage"]
        assert any("O text." in c["text"] for c in chunks)

    def test_text_before_the_first_heading_becomes_an_introduction_chunk(self):
        chunks = data_ingest.chunk_document("x.md", "# T\nIntro paragraph that matters.\n\n## Overview\nO text.\n")
        assert [c["section"] for c in chunks] == ["Introduction", "Overview"]
        assert "Intro paragraph that matters." in chunks[0]["text"]
        assert not any(c["text"].startswith("# T") for c in chunks)  # the title line is not content

    def test_a_file_with_no_section_headings_is_kept_not_dropped(self):
        chunks = data_ingest.chunk_document("x.md", "# T\nJust a paragraph with real dosing info.\n")
        assert [c["section"] for c in chunks] == ["Introduction"]
        assert "real dosing info" in chunks[0]["text"]

    def test_a_file_with_no_title_and_no_headings_is_kept_too(self):
        chunks = data_ingest.chunk_document("plain.md", "plain text only\n")
        assert [c["section"] for c in chunks] == ["Introduction"]
        assert chunks[0]["title"] == "plain.md"

    def test_a_triple_hash_subheading_stays_inside_its_section(self):
        chunks = data_ingest.chunk_document("x.md", "# T\n## Dosage\nAdults:\n### Children\nask a doctor\n")
        assert [c["section"] for c in chunks] == ["Dosage"]
        assert "### Children" in chunks[0]["text"] and "ask a doctor" in chunks[0]["text"]

    def test_a_title_only_file_has_nothing_to_keep(self):
        assert data_ingest.chunk_document("x.md", "# Only A Title\n") == []

    def test_the_standard_layout_is_unchanged(self):
        chunks = data_ingest.chunk_document("sample.md", SAMPLE_DOC)
        assert [c["section"] for c in chunks] == ["Overview", "Dosage", "Common Side Effects", "Warnings"]


class TestBuildIndex:
    def test_builds_a_real_index_with_a_mocked_embedding_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data_ingest, "CHROMA_DIR", tmp_path / "chroma_db")
        monkeypatch.setattr(data_ingest, "META_PATH", tmp_path / "kb_meta.json")

        def fake_embed(model, input):
            return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3, 0.4]])

        monkeypatch.setattr(data_ingest._client, "embed", fake_embed)

        chunks = data_ingest.build_index()

        # Real chunking: 10 knowledge-base files x 4 sections each.
        assert len(chunks) == 40
        assert all("embedding" in c for c in chunks)

        meta = json.loads(data_ingest.META_PATH.read_text())
        assert meta["content_hash"] == data_ingest.compute_content_hash()
        assert meta["embed_model"] == data_ingest.config.EMBED_MODEL

        # Real Chroma write, readable back from the same temp directory.
        client = chromadb.PersistentClient(path=str(data_ingest.CHROMA_DIR))
        collection = client.get_collection(data_ingest.COLLECTION_NAME)
        assert collection.count() == 40

    def test_rebuilding_drops_the_old_collection_first(self, tmp_path, monkeypatch):
        # A chunk from a since-edited/removed document must never linger.
        monkeypatch.setattr(data_ingest, "CHROMA_DIR", tmp_path / "chroma_db")
        monkeypatch.setattr(data_ingest, "META_PATH", tmp_path / "kb_meta.json")
        monkeypatch.setattr(
            data_ingest._client, "embed", lambda model, input: SimpleNamespace(embeddings=[[0.1, 0.2, 0.3, 0.4]])
        )

        data_ingest.build_index()
        data_ingest.build_index()  # must not raise or duplicate on a second run

        client = chromadb.PersistentClient(path=str(data_ingest.CHROMA_DIR))
        collection = client.get_collection(data_ingest.COLLECTION_NAME)
        assert collection.count() == 40
