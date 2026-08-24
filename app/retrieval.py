"""Vector search over the ingested knowledge base — embeds a query and finds
the most similar chunks via a persistent Chroma collection (see
data_ingest.py for how it's built).
"""

import json
import re

import chromadb
import ollama

from app import config, data_ingest

_client = ollama.Client(host=config.OLLAMA_HOST)

MIN_SIMILARITY = 0.63

MG_RE = re.compile(r"\d+\s*mg", re.IGNORECASE)
STRENGTH_BOOST = 0.2


def _mg_strengths(text: str) -> set[str]:
    return {m.replace(" ", "").lower() for m in MG_RE.findall(text)}


def _load_collection():
    """Loads the cached Chroma collection, rebuilding it if it's missing OR
    stale — the sidecar content_hash is compared against a fresh hash of the
    current knowledge-base files + embedding model (see
    data_ingest.compute_content_hash), so editing a knowledge_base/*.md file
    and restarting the app picks up the change automatically instead of
    silently serving outdated embeddings."""
    if data_ingest.META_PATH.exists():
        meta = json.loads(data_ingest.META_PATH.read_text())
        if meta.get("content_hash") == data_ingest.compute_content_hash():
            client = chromadb.PersistentClient(path=str(data_ingest.CHROMA_DIR))
            try:
                return client.get_collection(data_ingest.COLLECTION_NAME)
            except Exception:
                print("[RAG] cache metadata present but collection missing — rebuilding index")
        else:
            print("[RAG] knowledge base changed since last ingest — rebuilding index")
    else:
        print("[RAG] no cached index found — building one")
    data_ingest.build_index()
    return chromadb.PersistentClient(path=str(data_ingest.CHROMA_DIR)).get_collection(data_ingest.COLLECTION_NAME)


_collection = _load_collection()


def search(query: str, top_k: int = 3) -> list[dict]:
    """Returns up to top_k chunks most relevant to query, each as
    {source, section, text, score}, filtered to a minimum similarity so an
    unrelated query returns nothing rather than forcing irrelevant chunks
    into the model's context.

    Pure semantic similarity alone doesn't reliably distinguish exact dosage
    strengths — "paracetamol 500mg" scored fev-002 (650mg) above fev-001
    (500mg) in testing, since both chunks are otherwise near-identical
    paracetamol dosage text and the embedding model doesn't weight the
    number heavily. A small keyword boost for an exact "NNNmg" match between
    the query and a chunk's product title (a lightweight hybrid
    keyword+semantic technique) fixes this specific, verified failure mode
    without a full reranker — fetching more than top_k from Chroma before
    applying it means a strength-matching chunk can still be pulled back
    into the final top_k even if its raw vector similarity alone wouldn't
    have ranked it there."""
    if _collection.count() == 0:
        return []

    response = _client.embed(model=config.EMBED_MODEL, input=query)
    query_vector = response.embeddings[0]
    query_strengths = _mg_strengths(query)

    fetch_n = min(max(top_k * 3, 10), _collection.count())
    result = _collection.query(query_embeddings=[query_vector], n_results=fetch_n)

    scored = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        score = 1 - distance  # collection is cosine-space: distance = 1 - cosine_similarity
        if query_strengths and query_strengths & _mg_strengths(meta["title"]):
            score = min(score + STRENGTH_BOOST, 1.0)
        scored.append({"product": meta["title"], "source": meta["source"], "section": meta["section"], "text": doc, "score": score})
    scored.sort(key=lambda c: c["score"], reverse=True)

    return [
        {**c, "score": round(c["score"], 3)}
        for c in scored[:top_k]
        if c["score"] >= MIN_SIMILARITY
    ]
