"""Vector search over the ingested knowledge base — embeds a query and finds
the most similar chunks via a persistent Chroma collection (see
data_ingest.py for how it's built).
"""

import json
import re
from pathlib import Path

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


# Words that happen to appear in exactly one product title but are ordinary
# English in a query, so they must not by themselves pick a product: "what
# pain relief is there for fever" is not a question about "Cold & Flu
# Multi-Symptom Relief". Kept to the one word actually observed misfiring —
# "extra"/"strength" are deliberately NOT here, they are how "extra strength
# paracetamol" finds Paracetamol Extra Strength 650mg.
_GENERIC_ANCHOR_WORDS = {"relief"}

# A question about a child must not be narrowed to an adult-only product.
_CHILD_QUERY_RE = re.compile(
    r"\b(?:child|children|kid|kids|baby|babies|infant|toddler|toddlers|pediatric|paediatric"
    r"|\d+[- ]?(?:year|yr|month)s?[- ]?old)\b"
)


def _ingredient_owners() -> dict[str, set[str]]:
    """Each 5+ letter word of every product's active_ingredient -> the set of
    product titles containing it. Titles alone don't capture ingredients:
    "Children's Cold & Cough Syrup" contains dextromethorphan but its title
    never says so, while "Cough Suppressant Syrup (Dextromethorphan)" does."""
    catalog = json.loads((Path(__file__).parent / "data" / "catalog.json").read_text())
    owners: dict[str, set[str]] = {}
    for product in catalog:
        for word in TITLE_WORD_RE.findall(product["active_ingredient"].lower()):
            if len(word) >= 5:
                owners.setdefault(word, set()).add(product["name"])
    return owners


_ingredient_owners_by_word = _ingredient_owners()


def _title_words_by_title(collection) -> dict[str, set[str]]:
    """Every 5+ letter word of each product title (NOT just the distinctive
    ones), for _best_matching_title's overlap count."""
    titles = {m["title"] for m in collection.get()["metadatas"]}
    return {title: {w for w in TITLE_WORD_RE.findall(title.lower()) if len(w) >= 5} for title in titles}


def _best_matching_title(query_lower: str) -> str | None:
    """The product whose title overlaps the query MOST, if that is a unique
    winner with at least two matching words — for products that have no
    distinctive title word of their own (fev-001 "Paracetamol 500mg Tablets",
    fev-004 "Children's Paracetamol Syrup", col-006 "Children's Cold & Cough
    Syrup"), which _title_mentioned can therefore never recognize. A question
    about a child counts toward the pediatric products ("children" in the
    title) even when it says "child"/"kid"/"5 year old", since that is the
    whole way the query names them."""
    query_words = set(TITLE_WORD_RE.findall(query_lower))
    child_query = bool(_CHILD_QUERY_RE.search(query_lower)) and "children" not in query_words
    scores = {
        title: len(words & query_words) + (1 if child_query and "children" in title.lower() else 0)
        for title, words in _title_words.items()
    }
    best = max(scores.values(), default=0)
    winners = [title for title, score in scores.items() if score == best]
    return winners[0] if best >= 2 and len(winners) == 1 else None


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
    matched_titles = {
        title for title, words in _distinctive_words.items() if (words - _GENERIC_ANCHOR_WORDS) & query_words
    }
    if len(matched_titles) != 1:
        return None
    target = next(iter(matched_titles))

    # Real bug: narrowing to the ONE product a distinctive title word names
    # silently discarded every other product the query also pointed at.
    # "paracetamol vs ibuprofen which is better for fever" returned only
    # ibuprofen; "dextromethorphan dosage for children" returned only the
    # ADULT cough syrup's dosage, because only its title says dextromethorphan
    # while the children's syrup contains it too. Two ways the query can be
    # about more than the target:
    #  (a) it names an ingredient the target does NOT contain (another product
    #      is in play — "paracetamol" in a question matched to ibuprofen);
    #  (b) the only thing that matched the target is an ingredient other
    #      products share, so nothing singles THIS one out ("dextromethorphan"
    #      alone). A distinctive non-ingredient word ("extra strength") still
    #      narrows even when the query also says "paracetamol".
    for word in query_words:
        owners = _ingredient_owners_by_word.get(word)
        if owners and target not in owners:
            return None
    anchors = (_distinctive_words[target] - _GENERIC_ANCHOR_WORDS) & query_words
    if all(len(_ingredient_owners_by_word.get(word, ())) > 1 for word in anchors):
        return None

    # Same principle for age: never narrow a question about a child down to an
    # adult-only product's chunks.
    if _CHILD_QUERY_RE.search(query_lower) and "children" not in target.lower():
        return None
    return target


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
_title_words = _title_words_by_title(_collection)


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
    # Real bug: Ollama returns no embedding at all for an empty string, so
    # response.embeddings[0] raised IndexError — and this model is known to
    # pass empty-string tool args (see check_order_status's docstring), which
    # surfaced to it as "Error calling lookup_medicine_info: list index out
    # of range" instead of the clean "no information found" result.
    if not query.strip():
        return []

    response = _client.embed(model=config.EMBED_MODEL, input=query)
    if not response.embeddings:
        return []
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

    best_title = _best_matching_title(query_lower)
    scored = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        score = 1 - distance  # collection is cosine-space: distance = 1 - cosine_similarity
        if query_strengths and query_strengths & _mg_strengths(meta["title"]):
            score = min(score + STRENGTH_BOOST, 1.0)
        section_keywords = SECTION_KEYWORDS.get(meta["section"], ())
        # Real bug: this used to require _title_mentioned alone, which is never
        # true for a product with no distinctive title word, so for fev-001/
        # fev-004/col-006 the asked-for section was never boosted and ANOTHER
        # product's chunk won: "children's paracetamol dosing" returned the
        # children's syrup Overview plus the ADULT 650mg product's Dosage;
        # "children's cold and cough syrup dosage" returned the paracetamol
        # syrup's Dosage; "paracetamol 500mg dosage" returned no Dosage at all.
        product_named = (
            _title_mentioned(query_lower, meta["title"])
            or bool(query_strengths and query_strengths & _mg_strengths(meta["title"]))
            or meta["title"] == best_title
        )
        if any(kw in query_lower for kw in section_keywords) and product_named:
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
