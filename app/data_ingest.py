"""Document loading, chunking, and embedding for the medicine-info knowledge
base. Run standalone with `python -m app.data_ingest` to (re)build the index
independently of starting the app — see retrieval.py for how it's consumed.
"""

import hashlib
import json
import re
from pathlib import Path

import ollama

from app import config

KB_DIR = Path(__file__).parent / "data" / "knowledge_base"
INDEX_PATH = Path(__file__).parent / "data" / "kb_index.json"

_client = ollama.Client(host=config.OLLAMA_HOST)


def load_documents() -> list[tuple[str, str]]:
    """Returns [(filename, full_text), ...] for every .md file in the
    knowledge base, sorted for a deterministic ingestion order."""
    return [(path.name, path.read_text()) for path in sorted(KB_DIR.glob("*.md"))]


def compute_content_hash() -> str:
    """Fingerprints the current knowledge-base source files plus the
    embedding model name — if either changes, cached embeddings are stale
    and need rebuilding. This is what makes the cache in retrieval.py
    self-invalidating instead of silently serving outdated data after a
    knowledge-base edit."""
    hasher = hashlib.sha256()
    for filename, text in load_documents():
        hasher.update(filename.encode())
        hasher.update(text.encode())
    hasher.update(config.EMBED_MODEL.encode())
    return hasher.hexdigest()


def chunk_document(filename: str, text: str) -> list[dict]:
    """Splits a document on '## ' section headers. This is a structured,
    deterministic chunking strategy that fits our uniformly-templated docs
    (Overview / Dosage / Side Effects / Warnings) — fixed-size or token-based
    chunking would be the right call instead for unstructured long-form text,
    where section boundaries aren't meaningful or don't exist."""
    title_match = re.match(r"#\s+(.+)", text.strip())
    doc_title = title_match.group(1).strip() if title_match else filename

    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or not section.startswith("##"):
            continue
        header_match = re.match(r"##\s+(.+)", section)
        section_title = header_match.group(1).strip() if header_match else "Overview"
        chunks.append({
            "source": filename,
            "title": doc_title,
            "section": section_title,
            "text": section,
        })
    return chunks


def build_index() -> list[dict]:
    """Chunks every knowledge-base document, embeds each chunk with the local
    Ollama embedding model, and writes/returns the resulting chunks (each
    chunk dict gains an "embedding" key: a list of floats). The cache file
    also stores a content_hash so retrieval.py can detect when the source
    docs or embedding model have changed and rebuild automatically."""
    all_chunks = []
    for filename, text in load_documents():
        all_chunks.extend(chunk_document(filename, text))

    for chunk in all_chunks:
        response = _client.embed(model=config.EMBED_MODEL, input=chunk["text"])
        chunk["embedding"] = response.embeddings[0]

    INDEX_PATH.write_text(json.dumps({
        "content_hash": compute_content_hash(),
        "embed_model": config.EMBED_MODEL,
        "chunks": all_chunks,
    }))
    return all_chunks


if __name__ == "__main__":
    index = build_index()
    sources = sorted({c["source"] for c in index})
    print(f"Ingested {len(sources)} documents -> {len(index)} chunks")
    print(f"Embedding model: {config.EMBED_MODEL}")
    print(f"Wrote index to {INDEX_PATH}")
