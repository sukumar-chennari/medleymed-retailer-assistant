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
#
# "Distinctive" is checked against the whole catalog, not just the one
# title in isolation — once the catalog grew to include col-005/col-006
# (both cough/cold syrups), "cough" and "syrup" stopped being unique to
# col-004's title. "dosage for cough suppressant syrup" then boosted all
# three products' Dosage sections equally and lost the actual match to
# col-005/col-006 on raw semantic similarity — matching on any *shared*
# word is worse than not boosting at all, since it actively promotes the
# wrong products instead of just failing to promote the right one.
SECTION_BOOST = 0.15
SECTION_KEYWORDS = {
    "Dosage": ("dosage", "dose", "dosing", "how much", "how many"),
    "Common Side Effects": ("side effect", "side-effect"),
    "Warnings": ("warning", "caution", "contraindication"),
}
TITLE_WORD_RE = re.compile(r"[a-z]+")


def _mg_strengths(text: str) -> set[str]:
    return {m.replace(" ", "").lower() for m in MG_RE.findall(text)}


def _distinctive_words_by_title(collection) -> dict[str, set[str]]:
    """For every product title in the corpus, the 5+ letter words in that
    title that appear in NO other title — the only words safe to use as a
    per-product anchor. Computed once at load time from the collection's
    own metadata, so it stays correct as the catalog grows without needing
    a hardcoded word list."""
    titles = {m["title"] for m in collection.get()["metadatas"]}
    word_to_titles: dict[str, set[str]] = {}
    for title in titles:
        for word in {w for w in TITLE_WORD_RE.findall(title.lower()) if len(w) >= 5}:
            word_to_titles.setdefault(word, set()).add(title)
    return {
        title: {w for w, owning_titles in word_to_titles.items() if owning_titles == {title}}
        for title in titles
    }


def _title_mentioned(query_lower: str, title: str) -> bool:
    """True if the query names this specific product — shares a word with
    its title that's distinctive across the whole catalog (see
    _distinctive_words_by_title), e.g. "suppressant" for "Cough Suppressant
    Syrup (Dextromethorphan)" but not "cough" or "syrup" once other cough/
    cold syrups exist too."""
    query_words = set(TITLE_WORD_RE.findall(query_lower))
    return bool(_distinctive_words.get(title, set()) & query_words)


def _identify_single_product(query_lower: str) -> str | None:
    """Returns the one product title the query names, if a distinctive
    word identifies exactly one — used to filter results down to that
    product before they ever reach the model. This isn't optional
    polish: the system prompt already instructs the model to "only use
    results whose product matches" what was asked and ignore the rest,
    but it doesn't reliably follow that on its own — "can I take
    ibuprofen for a fever" retrieved both Ibuprofen and Paracetamol
    chunks, and the model cited both, recommending paracetamol as an
    unprompted "alternative" the user never asked about. Filtering the
    candidate set itself removes the chance for that regardless of
    whether the model would have honored the instruction that turn."""
    query_words = set(TITLE_WORD_RE.findall(query_lower))
    matched_titles = {title for title, words in _distinctive_words.items() if words & query_words}
    return next(iter(matched_titles)) if len(matched_titles) == 1 else None


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
_distinctive_words = _distinctive_words_by_title(_collection)


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

    # A boost can only re-rank a chunk that's already in this candidate
    # pool — it can't rescue one Chroma's raw nearest-neighbor search
    # didn't return at all. "col-004" Dosage ranked outside the old
    # top_k*3 window once the catalog grew to 10 products (more inter-
    # product semantic competition), so its keyword boost never got a
    # chance to apply. The corpus is still small (dozens of chunks, not
    # thousands), so fetching most of it unconditionally is cheap and
    # removes this whole class of "boost arrived too late" failure.
    fetch_n = min(30, _collection.count())
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

    target_product = _identify_single_product(query_lower)
    if target_product:
        scored = [c for c in scored if c["product"] == target_product] or scored

    scored.sort(key=lambda c: c["score"], reverse=True)

    return [
        {**c, "score": round(c["score"], 3)}
        for c in scored[:top_k]
        if c["score"] >= MIN_SIMILARITY
    ]
