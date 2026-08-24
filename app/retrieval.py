"""Vector search over the ingested knowledge base — embeds a query and finds
the most similar chunks by cosine similarity. Plain Python, no vector-DB
dependency: the corpus is ~30 chunks, which doesn't need one.
"""

import json
import math
import re

import ollama

from app import config, data_ingest

_client = ollama.Client(host=config.OLLAMA_HOST)

MIN_SIMILARITY = 0.63

MG_RE = re.compile(r"\d+\s*mg", re.IGNORECASE)
STRENGTH_BOOST = 0.2


def _mg_strengths(text: str) -> set[str]:
    return {m.replace(" ", "").lower() for m in MG_RE.findall(text)}


def _load_index() -> list[dict]:
    """Loads the cached index, rebuilding it if it's missing OR stale — the
    cache's stored content_hash is compared against a fresh hash of the
    current knowledge-base files + embedding model (see
    data_ingest.compute_content_hash), so editing a knowledge_base/*.md file
    and restarting the app picks up the change automatically instead of
    silently serving outdated embeddings."""
    if data_ingest.INDEX_PATH.exists():
        cached = json.loads(data_ingest.INDEX_PATH.read_text())
        if cached.get("content_hash") == data_ingest.compute_content_hash():
            return cached["chunks"]
        print("[RAG] knowledge base changed since last ingest — rebuilding index")
    else:
        print("[RAG] no cached index found — building one")
    return data_ingest.build_index()


_index = _load_index()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
    without a full reranker."""
    response = _client.embed(model=config.EMBED_MODEL, input=query)
    query_vector = response.embeddings[0]
    query_strengths = _mg_strengths(query)

    scored = []
    for chunk in _index:
        score = _cosine_similarity(query_vector, chunk["embedding"])
        if query_strengths and query_strengths & _mg_strengths(chunk["title"]):
            score = min(score + STRENGTH_BOOST, 1.0)
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda c: c["score"], reverse=True)

    return [
        {
            "product": c["title"],
            "source": c["source"],
            "section": c["section"],
            "text": c["text"],
            "score": round(c["score"], 3),
        }
        for c in scored[:top_k]
        if c["score"] >= MIN_SIMILARITY
    ]
