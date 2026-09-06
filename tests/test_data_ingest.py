"""Regression tests for the pure parts of app/data_ingest.py: document
loading and the header-based chunking strategy. Deliberately excludes
build_index, which needs a live Ollama embedding call and a real Chroma
write — that path is already exercised indirectly by CI (see
.github/workflows/tests.yml's comment: importing app.tools triggers
retrieval.py's cold-start rebuild whenever no cached index exists yet).
"""

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
