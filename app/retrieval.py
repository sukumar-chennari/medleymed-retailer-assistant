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

# A second hybrid keyword+semantic boost, same idea as STRENGTH_BOOST but for
# section intent rather than dosage strength: "dosage for cough suppressant
# syrup" scored col-004's own Dosage section (0.606) *below* its Overview
# (0.672) and Warnings (0.654) sections — all three are short, topically
# similar paragraphs about the same product, and nothing in the query itself
# gives the embedding a strong reason to prefer Dosage specifically. Since
# the query already names which section it wants in plain words, a keyword
# match against the chunk's own section name is exactly the same kind of
# targeted, verified fix as the strength boost, not a general reranker.
#
# Gating this on the query ALSO naming the product (sharing a distinctive
# word with the chunk's own title) is load-bearing, not optional — a first
# version boosted every chunk whose section matched the keyword regardless
# of product, which (a) undid the fix it was meant to make, since every
# *other* product's Dosage section got boosted too and outranked the right
# one, and (b) turned "dosage for amoxicillin" into a false positive, since
# a generic "dosage for X" query resembles the Dosage section of *any*
# product almost equally on pure semantics — there's nothing to boost
# without a real per-product anchor the way STRENGTH_BOOST has one.
SECTION_BOOST = 0.15
SECTION_KEYWORDS = {
    "Dosage": ("dosage", "dose", "dosing", "how much", "how many"),
    "Common Side Effects": ("side effect", "side-effect"),
    "Warnings": ("warning", "caution", "contraindication"),
}
TITLE_WORD_RE = re.compile(r"[a-z]+")


def _mg_strengths(text: str) -> set[str]:
    return {m.replace(" ", "").lower() for m in MG_RE.findall(text)}


def _title_mentioned(query_lower: str, title: str) -> bool:
    """True if the query names this specific product — shares a
    distinctive (5+ letter) word with its title, e.g. "cough"/"suppressant"/
    "syrup" for "Cough Suppressant Syrup (Dextromethorphan)". Short words are
    excluded for the same reason tools.py's fuzzy symptom matching excludes
    them: too many unrelated collisions (a generic word like "syrup" alone
    would be fine, but this keeps the bar consistent with the rest of the
    codebase's hybrid-matching functions)."""
    title_words = TITLE_WORD_RE.findall(title.lower())
    return any(len(w) >= 5 and w in query_lower for w in title_words)


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
    have ranked it there. SECTION_BOOST applies the same idea to section
    intent (see its own docstring above)."""
    if _collection.count() == 0:
        return []

    response = _client.embed(model=config.EMBED_MODEL, input=query)
    query_vector = response.embeddings[0]
    query_strengths = _mg_strengths(query)
    query_lower = query.lower()

    fetch_n = min(max(top_k * 3, 10), _collection.count())
    result = _collection.query(query_embeddings=[query_vector], n_results=fetch_n)

    scored = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        score = 1 - distance  # collection is cosine-space: distance = 1 - cosine_similarity
        if query_strengths and query_strengths & _mg_strengths(meta["title"]):
            score = min(score + STRENGTH_BOOST, 1.0)
        section_keywords = SECTION_KEYWORDS.get(meta["section"], ())
        if any(kw in query_lower for kw in section_keywords) and _title_mentioned(query_lower, meta["title"]):
            score = min(score + SECTION_BOOST, 1.0)
        scored.append({"product": meta["title"], "source": meta["source"], "section": meta["section"], "text": doc, "score": score})
    scored.sort(key=lambda c: c["score"], reverse=True)

    return [
        {**c, "score": round(c["score"], 3)}
        for c in scored[:top_k]
        if c["score"] >= MIN_SIMILARITY
    ]
