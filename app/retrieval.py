"""Vector search over the ingested knowledge base — embeds a query and finds
the most similar chunks by cosine similarity. Plain Python, no vector-DB
dependency: the corpus is ~30 chunks, which doesn't need one.
"""

import json
import math

import ollama

from app import config, data_ingest

_client = ollama.Client(host=config.OLLAMA_HOST)

MIN_SIMILARITY = 0.55


def _load_index() -> list[dict]:
    if not data_ingest.INDEX_PATH.exists():
        return data_ingest.build_index()
    return json.loads(data_ingest.INDEX_PATH.read_text())


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
    into the model's context."""
    response = _client.embed(model=config.EMBED_MODEL, input=query)
    query_vector = response.embeddings[0]

    scored = [
        {**chunk, "score": _cosine_similarity(query_vector, chunk["embedding"])}
        for chunk in _index
    ]
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
