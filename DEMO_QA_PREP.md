# MedleyMed Retailer Assistant — Demo Q&A Prep

A question bank for defending the architecture of this project in a live demo.
Every answer is grounded in the actual code in this repo — file names, exact
numbers, and real bugs found and fixed — not generic RAG theory. When you
answer, it's fine (good, even) to reference the specific file: it signals you
actually built this rather than describing a pattern you read about.

**One-paragraph pitch**, in case you're asked to open with it: *"MedleyMed
Retailer Assistant is a scoped fever/cold OTC medicine assistant — an agentic
chatbot that takes symptoms or a photo of a medicine label, recommends a
product from a fixed 10-item catalog, and places a demo order with address and
email confirmation. It runs entirely on free, local resources: Ollama
(`llama3.2`) for conversation and tool-calling, a local Chroma vector store
for a small RAG knowledge base, and the free Gemini API only for reading
photos. It's deliberately narrow in scope but the engineering focus was
reliability — closing real hallucination and guardrail gaps found through
live testing, not just building a happy-path demo."*

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Corpus Collection](#2-data-corpus-collection)
3. [Chunking Strategy](#3-chunking-strategy)
4. [Vector Database & Embeddings](#4-vector-database--embeddings)
5. [Retrieval Strategy](#5-retrieval-strategy)
6. [Re-ranking / Hybrid Search](#6-re-ranking--hybrid-search)
7. [LLM & Tool-Calling Architecture](#7-llm--tool-calling-architecture)
8. [Guardrails](#8-guardrails)
9. [Hallucinations — Real Bugs Found and Fixed](#9-hallucinations--real-bugs-found-and-fixed)
10. [Evaluation](#10-evaluation)
11. [Persistence / Data Layer](#11-persistence--data-layer)
12. [Hard / Adversarial Questions](#12-hard--adversarial-questions)

---

## 1. Architecture Overview

**Q: Walk me through the request lifecycle — what happens when a user types a message?**
A: `main.py`'s `/api/chat` route first checks a priority-ordered chain of
*deterministic* session states — is there a pending clarifying question, a
pending address confirmation, a pending email request, a just-shown product
list awaiting a bare selection like "2"? If any of those match, the reply is
generated in plain Python with zero LLM involvement. Only if none of those
match does it fall through to `agent.run_turn()`, which builds the message
payload (injecting a deterministic catalog hint if the text mentions a
symptom) and hands it to a LangChain agent (`langchain.agents.create_agent`,
built on LangGraph) wired with the 6 tools and a custom guardrail middleware.
The agent loops through any tool calls the model makes (capped via a
recursion limit) until it produces a plain-text reply; that reply — already
passed through the guardrail middleware's `after_agent` hook — goes back to
the user.

**Q: Why did you split logic between deterministic Python and the LLM at all — why not let the model handle everything?**
A: Because a 3B local model measurably cannot be trusted with certain classes
of decisions across multiple turns — remembering it already asked a
question, not fabricating an answer to a question it never asked, not
switching context without basis. Every one of those failure modes was
observed in live testing (see section 9). The rule I converged on: **if a
step is safety-relevant, state-relevant, or has exactly one correct answer
derivable from data we already have, don't ask the model — compute it.**
Only genuinely open-ended judgment (which product fits a symptom, how to
phrase a reply) is left to the LLM.

**Q: What are the main modules and what does each own?**
A:
- `agent.py` — system prompt, tool schema, the LangChain agent (`create_agent`) and its guardrail middleware
- `guardrails.py` — all hallucination/leak/pleasantry safety-net checks, applied to the model's output *after* it responds
- `tools.py` — tool implementations (symptom lookup, order placement, RAG lookup)
- `data_ingest.py` — loads, chunks, embeds, and upserts the knowledge base into Chroma
- `retrieval.py` — vector search + re-ranking over the Chroma collection
- `store.py` — SQLite-backed persistence (orders, users, session state)
- `main.py` — FastAPI routes and the deterministic pre-LLM dispatch chain

**Q: Is this using LangChain, LlamaIndex, or an agent framework?**
A: LangChain, yes — specifically its newer `create_agent` API
(`langchain.agents.create_agent`), which is built on LangGraph under the
hood, wired to `ChatOllama` (the LangChain integration for local Ollama
models) and a custom `AgentMiddleware` subclass that carries every guardrail
in this project. Not LlamaIndex — the RAG side is Chroma's own client
directly (see section 4), since LlamaIndex's value-add is mostly around
indexing/retrieval abstractions I'd already built by hand and verified with
an eval suite; adding it wouldn't have replaced meaningful custom code.

**Q: Did you start with LangChain, or add it later — and why?**
A: Added later, deliberately. The project started with a hand-rolled loop
over the raw Ollama chat API (`ollama.Client().chat(..., tools=[...])`) —
at 6 tools and one model, that was genuinely simpler to reason about and
debug than adopting a framework upfront, and it's what let me find and fix
every hallucination in section 9 by stepping through plain Python. I
migrated to LangChain afterward, once the guardrail logic was mature and
well-tested, specifically to demonstrate real framework usage rather than
leave it out entirely. The migration was the harder direction to do safely
— porting *into* a framework after guardrails already exist, rather than
building a framework-native agent that never had to reproduce them.

**Q: How did you avoid losing all the guardrail fixes from section 9 when you migrated to LangChain?**
A: By finding the two hook points in `create_agent`'s middleware API that
correspond exactly to where the old manual loop intervened, and moving each
guardrail there without changing its logic. `AgentMiddleware.wrap_tool_call`
intercepts a tool call *before* it executes — this is where the
premature-`start_order` block, the `lookup_symptom` category-switch
correction, the info-question redirect, and the `decline_out_of_scope`
reversal all live now, each returning a synthetic `ToolMessage` to
short-circuit the real tool when a call should be blocked or redirected, or
calling the real tool via `handler(request)` and post-processing its result
when it shouldn't. `AgentMiddleware.after_agent` fires once the model
produces its final (non-tool-calling) message — this is where leaked-tool
detection, unverified-completion checks, and the deterministic
order-confirmation/deferred-order templates apply, by returning a
same-`id` `AIMessage` that LangGraph merges in place to override the
model's own text. Every guardrail from section 9 was re-verified live
against the exact scenario that originally caught it, post-migration, before
this was considered done — not just re-read for logical equivalence.

**Q: Doesn't rebuilding the agent/tools/middleware on every single turn add overhead?**
A: Some, but it's pure Python object construction — no network calls, no
model loading — so it's on the order of microseconds, immaterial next to a
local LLM inference call. The reason it's rebuilt per turn rather than
cached is that both the tools (`start_order` needs `session_id` closed
over) and the middleware (needs this turn's `user_text`,
`symptom_lookup_grounded`, and a fresh mutable `turn_state` dict for
tracking real order/email events) are turn-specific — caching them would
mean either leaking one turn's context into another, or re-introducing
exactly the kind of shared mutable state bug this project spent most of its
effort eliminating elsewhere.

---

## 2. Data Corpus Collection

**Q: Where did the knowledge base content come from?**
A: `app/data/knowledge_base/` — 10 hand-written Markdown files, one per
catalog product (`fev-001.md`...`fev-004.md`, `col-001.md`...`col-006.md`).
Content is standard, widely-known OTC reference facts: typical adult dosing
intervals, commonly documented side effects, and standard label warnings
(don't exceed the max dose, consult a doctor if pregnant, etc.) — the kind of
information printed on an actual OTC medicine box.

**Q: Why hand-write the corpus instead of scraping real drug-label data (e.g. DailyMed, FDA)?**
A: Two reasons. First, scope: this catalog is 10 fictional-brand products
mapped onto real active ingredients (paracetamol, ibuprofen, cetirizine,
pseudoephedrine, dextromethorphan) — there's no single real-world label that
matches "Paracetamol Extra Strength 650mg" from this specific catalog, so a
real scrape would need normalization work disproportionate to a demo. Second,
correctness: hand-writing let me guarantee every document follows the exact
same section structure, which is what makes the chunking strategy (section 3)
reliable. In a production system I'd absolutely swap this for a real
pharmacological source with proper licensing.

**Q: Why one document per product instead of one big document, or one per symptom?**
A: Grounding precision. If "side effects of cetirizine" and "side effects of
pseudoephedrine" lived in the same document, a chunk boundary drawn in the
wrong place could blend both products' side effects into one retrieved
chunk — and I actually hit exactly this failure in testing (see section 9).
One document per product means every chunk's identity is unambiguous from
the start, before chunking even happens.

**Q: What's the total corpus size?**
A: 10 documents, 40 chunks after chunking (4 chunks per document — Overview,
Dosage, Common Side Effects, Warnings — since all 10 follow the identical
section template).

**Q: How do you keep the corpus in sync with the product catalog (`catalog.json`)?**
A: Manually, by convention — the knowledge base filename matches the catalog
product id (`fev-001.md` ↔ `fev-001` in `catalog.json`), and the doc's H1
title matches the catalog product name. At this scale that's an acceptable
manual contract; at real scale I'd generate a stub knowledge-base file
automatically whenever a catalog product is added, and fail CI if one is
missing.

**Q: Is there any PII or real patient data in the corpus?**
A: No — it's reference-only content (dosage/side-effects/warnings), not
records, cases, or anything patient-specific. No privacy concerns to design
around here.

---

## 3. Chunking Strategy

**Q: What chunking strategy did you use, and why?**
A: **Header-based structural chunking** — `data_ingest.chunk_document()`
splits each document on `## ` (H2) boundaries via regex
(`re.split(r"\n(?=## )", text)`), producing one chunk per section: Overview,
Dosage, Common Side Effects, Warnings. I chose this over fixed-size or
token-count chunking because the corpus is *uniformly templated* — every
document has the exact same four sections in the exact same order, so the
natural semantic boundaries (a dosage question should retrieve dosage text,
not warnings text) line up exactly with the structural boundaries. Fixed-size
chunking would be the right call for unstructured long-form text where
section boundaries don't exist or aren't consistent — that's not this corpus.

**Q: What's your chunk size — did you set a token limit?**
A: No fixed token limit — chunk size is whatever a natural section is (a
paragraph or two, typically 40-80 words). Since chunking follows document
structure rather than a size target, there's no risk of a chunk being cut
mid-sentence or mid-thought the way naive fixed-size chunking can produce.

**Q: What metadata do you attach to each chunk, and why?**
A: Three fields: `source` (filename, e.g. `fev-001.md`), `title` (the
document's H1 — the product's full name), and `section` (the H2 that
produced this chunk, e.g. "Dosage"). `title` is what enables the
dosage-strength re-ranking boost (section 6) and the "only trust results
about the medicine actually asked about" cross-check in the system prompt
(section 8) — it's the single most load-bearing piece of metadata in the
whole RAG pipeline.

**Q: Would this chunking strategy work if the corpus grew to hundreds of documents from different sources?**
A: Not as-is, no — header-based chunking assumes a uniform template, which
breaks the moment you ingest a document that isn't Overview/Dosage/Side
Effects/Warnings shaped (e.g. a long-form clinical monograph, a news
article, an FAQ page). At that scale I'd move to a hybrid: structural
chunking where a template exists, falling back to recursive
character/token-based chunking (e.g. LangChain's `RecursiveCharacterTextSplitter`
pattern — split on paragraph, then sentence, then character, with overlap)
for anything unstructured, plus per-source-type routing.

**Q: Any chunk overlap?**
A: None currently — since chunks are whole, disjoint sections, there's no
sentence context split across two chunks the way there would be with
fixed-size sliding-window chunking, so overlap wasn't needed to preserve
local context.

---

## 4. Vector Database & Embeddings

**Q: What vector database do you use?**
A: **Chroma**, in persistent local mode (`chromadb.PersistentClient`),
storing to `app/data/chroma_db/`. One collection, `medicine_kb`, configured
with `hnsw:space: "cosine"` so distance is cosine distance throughout.

**Q: Did you always use Chroma, or did this change?**
A: It changed. Originally I used a flat JSON file (`kb_index.json`) holding
each chunk plus its embedding vector, with cosine similarity computed by
hand in pure Python against every chunk (`retrieval.py`'s original
`_cosine_similarity`). That was a deliberate choice at first — the corpus is
only 32 chunks, so a full linear scan is microseconds and a vector DB would
have been pure overhead. I migrated to Chroma later once persistence became
a real requirement and it made sense to demonstrate the vector-DB concept
properly rather than the "good enough for 32 chunks" shortcut — same
retrieval interface and behavior, just backed by a real ANN index now.

**Q: Why Chroma specifically, and not Pinecone, Weaviate, FAISS, or pgvector?**
A: Free/local/no-server was the hard constraint for this whole project
(matches the choice of local Ollama over any paid API). Chroma runs
in-process with zero external server, persists to a local directory, and
has a clean Python client — no API keys, no network calls, no cost.
Pinecone/Weaviate are hosted-first (free tiers exist but add network
dependency and account setup); FAISS is a great local option too but has no
built-in metadata filtering or persistence layer, meaning I'd have to
build the same bookkeeping (source/section/title lookup) Chroma gives for
free; pgvector needs a running Postgres, which is more infrastructure than
this demo needs.

**Q: What embedding model do you use, and why?**
A: `nomic-embed-text`, served locally via Ollama. It's free (runs on the
same local Ollama instance as the chat model), produces 768-dimensional
embeddings, and is specifically trained/tuned for retrieval tasks (unlike a
general-purpose LLM's hidden states). No embedding API calls leave the
machine.

**Q: Are query embeddings and document embeddings generated the same way?**
A: Yes — same model, same `ollama.Client().embed()` call, both in
`retrieval.search()` (for the query) and `data_ingest.build_index()` (for
each chunk at ingestion time). Symmetric embedding is important for cosine
similarity to be meaningful; asymmetric embedding (e.g. a different model or
prompt template for queries vs. documents) would need re-calibrating the
similarity thresholds.

**Q: How do you keep the vector index in sync with the source Markdown files?**
A: Content-hash-based cache invalidation. `data_ingest.compute_content_hash()`
hashes every knowledge-base file's contents plus the embedding model name
into one SHA-256 digest. That hash is stored in a small sidecar file
(`app/data/kb_meta.json`) alongside the Chroma collection. On startup,
`retrieval._load_collection()` recomputes the hash fresh and compares it to
the stored one — if they differ (someone edited a `.md` file, or changed
`OLLAMA_EMBED_MODEL`), the collection is dropped and rebuilt automatically.
If they match, the existing collection loads instantly with no re-embedding.
This means editing the knowledge base and restarting the app "just works"
without a manual re-ingest step, while a normal restart with no changes
costs nothing.

**Q: What happens if the Chroma collection exists but is corrupted or partially built?**
A: `_load_collection()` wraps `client.get_collection()` in a broad
`try/except` — if the collection is missing (or fails to load for any
reason) despite the metadata file claiming it's current, it falls through to
rebuilding from scratch via `data_ingest.build_index()`. Belt-and-suspenders
against a half-written state, rather than crashing on startup.

**Q: Could you run this fully offline?**
A: For the RAG path and the chat model, yes — everything is local Ollama +
local Chroma. The only network dependency is the Gemini Vision API, used
solely for reading uploaded photo/prescription images (text-only symptom
chat works fully offline).

---

## 5. Retrieval Strategy

**Q: What's your retrieval algorithm?**
A: Embed the query with `nomic-embed-text`, query the Chroma collection for
the `fetch_n` nearest neighbors by cosine distance, apply the dosage-strength
re-ranking boost (section 6), sort by final score, then keep only results
scoring at or above `MIN_SIMILARITY` (0.63), truncated to `top_k` (default
3).

**Q: Why fetch more than `top_k` from the vector DB before filtering?**
A: `fetch_n = min(max(top_k * 3, 10), collection.count())` — I deliberately
over-fetch (at least 10, or 3x top_k) so the re-ranking boost has room to
work. If I only pulled the top 3 by raw cosine similarity and *then* boosted,
a chunk that's the right answer but ranks 4th or 5th on pure semantic
similarity (because the embedding model under-weights an exact dosage
number, see section 6) would never even be in the boost's candidate pool.
Over-fetching first, then boosting, then truncating is what lets the boost
actually change the final top-3.

**Q: Why 0.63 for `MIN_SIMILARITY`? Isn't that arbitrary?**
A: It's calibrated, not arbitrary — I measured real query scores. Relevant
in-scope queries scored roughly 0.65-0.81 cosine similarity against their
correct chunk; unrelated queries (e.g. "credit card refund policy",
"python programming help") scored roughly 0.40-0.48. 0.63 sits in the gap
between those two clusters, tuned specifically to reject unrelated queries
while accepting genuine ones. I originally had it at 0.3 (far too loose — an
unrelated "credit card refund policy" query returned 3 false-positive
results before I raised it), and the 12-query eval set (section 10) is what
caught that and let me verify the recalibration.

**Q: Why does an unrelated query returning nothing matter — what's the risk of a looser threshold?**
A: If the threshold is too loose, the model gets irrelevant chunks forced
into its context for an out-of-scope question and may compose an answer
grounded in the *wrong* text rather than admitting it has nothing — a
different flavor of hallucination than fabricating from pure imagination,
but just as unreliable. A hard cutoff means "no results" is itself a
meaningful, trustworthy signal the model is instructed to respect (see the
`lookup_medicine_info` tool description and the MEDICINE INFO system-prompt
section).

**Q: How many results do you return to the model, and how are they used?**
A: Up to `top_k=3`, each as `{product, source, section, text, score}`. The
system prompt instructs the model to answer *only* from the returned text,
discard any result whose `product` field doesn't match what the user
actually asked about (see section 8 — this closes a real cross-contamination
bug), and cite `(Source: fev-001.md § Dosage)` — both filename and section —
at the end of its answer.

**Q: Is retrieval used for anything besides the dedicated `lookup_medicine_info` tool?**
A: No — it's a single-purpose tool, deliberately kept separate from the
catalog/symptom-matching path (`tools.lookup_symptom`, which is plain keyword
matching against a fixed 10-item list, not vector search at all). Conflating
the two was actually a real, verified failure mode (see section 9) — an
info-style question like "side effects of cetirizine" was misrouted through
`lookup_symptom` (returning a bare cold-product list) instead of
`lookup_medicine_info` (returning grounded text), because "cetirizine" is
also a recognized cold keyword.

---

## 6. Re-ranking / Hybrid Search

**Q: Do you do any re-ranking, and why?**
A: Yes — a lightweight **hybrid keyword+semantic boost**, not a full
cross-encoder reranker. `retrieval.STRENGTH_BOOST = 0.2` is added to a
chunk's cosine score whenever the query and the chunk's product title share
an exact "NNNmg" dosage-strength match (matched via `MG_RE =
re.compile(r"\d+\s*mg")`).

**Q: What specific problem does the strength boost solve — walk me through the actual failure?**
A: Pure semantic similarity doesn't reliably distinguish exact numbers.
"Dosage for paracetamol 500mg" scored the **650mg** Extra Strength chunk
(fev-002) higher than the correct **500mg** chunk (fev-001) in testing —
both chunks are near-identical paracetamol dosage text, and the embedding
model doesn't weight the specific digit heavily enough to separate them on
meaning alone. The fix: after computing cosine similarity, check whether the
query's mg mention and the chunk's product-title mg mention match exactly;
if so, add +0.2 to that chunk's score (capped at 1.0). This is a small,
targeted hybrid technique, not a general-purpose reranker — it fixes one
specific, verified, reproducible failure mode rather than trying to solve
re-ranking in general.

**Q: Why a flat +0.2 bonus instead of a learned reranker (e.g. a cross-encoder model)?**
A: Proportionality. A cross-encoder reranker is the textbook right answer at
scale, but it's another model to run, another dependency, and meaningful
extra latency — for a corpus where the *entire* failure mode I could find
was "exact numeric strength gets under-weighted," a targeted keyword boost
fixes it completely with zero extra inference cost. I'd reach for a real
reranker if the corpus were large/diverse enough that semantic-only ranking
had more failure modes than this one specific, well-understood gap.

**Q: Could the strength boost ever hurt — push a wrong chunk above a right one?**
A: In principle, if two different products' titles happened to share the
same mg value with the same active ingredient the boost couldn't
distinguish them by strength alone — but at 10 products with no two sharing
both an active ingredient and a dosage strength, this hasn't occurred, and
the eval suite (section 10) would catch it if it started to as the catalog
grows.

**Q: Is there any re-ranking based on recency, popularity, or business logic (e.g. margin, stock)?**
A: No — purely relevance-based (semantic + the two hybrid boosts below).
That's a reasonable extension for a real e-commerce RAG system, but out of
scope for a demo where the "business logic" is a fixed 10-item catalog with
no stock or pricing tiers to optimize for.

**Q: You mentioned "the strength boost" — is there a second boost too?**
A: Yes — `SECTION_BOOST`, added after the golden eval (section 10) caught a
second real gap: "dosage for cough suppressant syrup" retrieved the right
product but lost to its own Overview/Warnings sections on pure semantics,
since all three read as similarly generic paragraphs about the same
product with nothing pointing the embedding at Dosage specifically. Same
hybrid pattern as the strength boost — the query already names which
section it wants in plain words ("dosage", "side effects", "warning"), so a
keyword match against the chunk's own section name resolves it. The
non-obvious part: this only fires when the query *also* names the specific
product (shares a distinctive word with the chunk's title) — without that
gate, a first version boosted every product's Dosage section
indiscriminately whenever a query said "dosage", which undid the fix
*and* turned "dosage for amoxicillin" into a new false positive, since a
generic "dosage for X" query resembles the Dosage section of almost any
product about equally well on pure semantics. Product-gating it closes
that; `STRENGTH_BOOST` gets the same kind of implicit product-anchor for
free, since an exact mg number is already fairly product-specific.

**Q: What actually happened when you scaled the catalog from 8 products to 10?**
A: Two real regressions, both caught by re-running the golden eval
immediately after adding col-005 (Guaifenesin, a wet-cough expectorant)
and col-006 (a children's cold/cough syrup) — a good demonstration of why
the eval exists at all, not just a one-time gate. First,
`_title_mentioned`'s "distinctive word" check was only ever distinctive
*within one title*, not across the catalog — once col-005 and col-006 both
shared "cough"/"syrup" with col-004's title, `SECTION_BOOST` fired for all
three on the same query and the right chunk lost. Fixed by computing
distinctiveness against every title in the corpus at load time (any word
appearing in more than one title is disqualified as an anchor), so
`SECTION_BOOST` still resolves to "suppressant" — genuinely unique to
col-004 — instead of the now-ambiguous shared words. Second, the boost
re-ranks only within a fetched candidate window (`top_k * 3` at the time);
with more products competing for the same semantic space, col-004's own
Dosage chunk fell outside that window entirely at `top_k=3`, so the boost
never got a chance to apply — fixed by widening the fetch to most of the
corpus unconditionally (still cheap at this scale) rather than tying the
window size to `top_k`.

**Q: Did scaling the catalog surface anything in the conversation layer too, not just retrieval?**
A: One real, safety-relevant bug: cough's dry/wet branching and the
fever/cold age branching are two independent dimensions in
`CLARIFYING_QUESTIONS`, and nothing composed them. "My child has a wet
cough" — or even "my child has a cough," asked the dry/wet question, then
answered "wet" a turn later — resolved through the cough branch alone and
recommended col-005, the *adult* expectorant, to a child. Fixed by
detecting an age qualifier alongside a cough qualifier and threading it
through: a pending cough clarification now carries an age suffix
(`"cough:child"`) that survives to the follow-up turn, and resolution
special-cases child+cough to route to col-006 (dry) or a pharmacist
referral (wet — there's no product for a child's wet cough at all) instead
of cough's normal adult branches. Verified all four combinations
(child/adult × same-message/cross-turn) individually before considering it
fixed.

---

## 7. LLM & Tool-Calling Architecture

**Q: What LLM are you using, and why that one?**
A: `llama3.2` (the 3B parameter variant), served locally via Ollama. Chosen
entirely for the "free resources only" constraint driving every architecture
decision in this project — no API costs, runs on a laptop CPU, and Ollama's
native tool-calling API (OpenAI-compatible `tools=[...]` schema) supports
everything the agent needs without any custom prompt-based tool-call parsing.

**Q: A 3B model is small — doesn't that hurt reliability? How did you deal with that?**
A: It genuinely does hurt reliability, and that's the central engineering
story of this project (section 9 has the specifics). The model reliably
handles single-turn judgment calls (which product fits a described symptom,
how to phrase a reply) but is unreliable at anything requiring multi-turn
memory, strict instruction-following under distraction, or judgment calls
with a safety consequence. My answer wasn't "use a bigger model" (that
breaks the free-resources constraint) — it was to make every
reliability-critical decision **deterministic Python instead of an LLM
judgment call**, and treat the LLM as the component you verify, never the
component you trust blindly.

**Q: What temperature do you run at, and why?**
A: 0.2 — low, favoring consistency over creative variation. This isn't a
creative-writing use case; for a scoped medical-adjacent assistant, "says
the same reasonable thing reliably" beats "occasionally says something more
interesting."

**Q: How many tools does the agent have, and what are they?**
A: Six: `lookup_symptom` (fixed catalog lookup by fever/cold keyword),
`get_saved_address`, `save_address`, `start_order` (defers to an
address-confirmation step, never completes synchronously), `lookup_medicine_info`
(the RAG tool), and `decline_out_of_scope`.

**Q: How does the tool-calling loop work?**
A: `agent.run_turn()` builds a LangChain agent per turn via `create_agent(
chat_model, tools=agent_tools, system_prompt=SYSTEM_PROMPT,
middleware=[guardrail_middleware])` and calls `.invoke({"messages": history})`.
Internally that's a LangGraph loop: call the model, check for tool calls, if
present dispatch each one (routed through the guardrail middleware's
`wrap_tool_call` hook first — see section 8) and feed its result back to the
model, repeat until a plain-text (non-tool-calling) message comes back. A
`recursion_limit` on the invoke call (`MAX_TOOL_ROUNDS * 2 + 4` — roughly two
graph steps per tool round, plus headroom) bounds this the same way the old
manual loop's `MAX_TOOL_ROUNDS = 6` cap did; hitting it raises
`GraphRecursionError`, caught in `run_turn` to return a generic fallback
reply instead of looping forever.

**Q: Why cap tool rounds at roughly 6 specifically?**
A: A real conversational turn rarely needs more than 2-3 tool calls even in
the more complex flows (e.g. lookup → start_order → done). 6 is generous
headroom above that while still bounding worst-case latency and preventing
an infinite loop if the model gets stuck repeatedly calling the same tool.

**Q: How does image input (a photo of a medicine label) get into this pipeline?**
A: Separately from the local LLM — `agent.describe_image()` sends the
uploaded image to the free Gemini API (vision), asking it to describe any
visible medicine name or active ingredient. That description is injected
into the conversation as `[Image analysis: ...]` text, then flows through
the exact same catalog-hint-injection and tool-calling path as if the user
had typed the symptom — the local model never sees raw image bytes.

**Q: Why use a different provider (Gemini) for vision instead of a local vision model?**
A: The free-resources constraint again, applied pragmatically — installing
and running a local vision-capable model was heavier (both in setup and in
inference time on a laptop CPU) than using Gemini's genuinely free API tier
for the one narrow task of "read text off a photo." Text conversation stays
100% local; vision is the one deliberate exception, and it's still free.

**Q: Is conversation history sent on every turn, or does the model only see the latest message?**
A: Full history — every prior user/assistant/tool message in the session is
sent alongside the system prompt on every call (`[{"role": "system", ...}] +
messages`). That's what lets `symptom_lookup_grounded` reasoning ("a category
was already established earlier this session") work, but it's also exactly
the mechanism that made deterministic state machines necessary — the model
re-reads the whole history each turn and doesn't reliably "remember" what it
already asked or already said unless the code enforces it structurally.

---

## 8. Guardrails

**Q: What's your overall guardrails philosophy?**
A: Every guardrail in `guardrails.py` exists because the small local model
proved unreliable at something specific, observed through live testing —
none of it is speculative "best practice" bolted on preemptively. And
critically: **every guardrail is a deterministic check on the model's actual
output or actual tool results, never a prompt-only instruction.** A prompt
saying "don't do X" is a request; a Python `if` statement checking for X is
a guarantee.

**Q: How do you keep the assistant in scope (fever/cold only)?**
A: Two layers. First, the system prompt explicitly instructs the model to
call a real `decline_out_of_scope` tool for anything else, rather than
answering off-topic requests itself. Second — and this is the layer that
actually matters — every tool that could act on a symptom
(`lookup_symptom`, `start_order`, `lookup_medicine_info`) only ever touches
the fixed 10-item catalog; there is no code path by which the assistant could
surface a product or answer that isn't in that catalog. The scope boundary
is enforced by *what the tools are physically capable of doing*, not by the
model's willingness to follow an instruction.

**Q: What's `symptom_lookup_grounded` and why does it exist?**
A: A cross-check that runs before trusting any `lookup_symptom` tool call:
is this call actually justified by (a) an image being present, (b) the
current message's own text classifying as fever/cold, or (c) a category
already established earlier in the session? If none of those hold, the call
is blocked and the turn is forced to a real decline — regardless of what the
model itself decided to do. This exists because adding the 6th tool
(`lookup_medicine_info`) measurably increased how often the model defaulted
to calling `lookup_symptom(symptom="fever")` for completely unrelated text
like "fix my code" — a 100% reproducible regression from raising the tool
count, verified by repeated testing, not guessed at.

**Q: How do you stop the model from claiming it did something it didn't?**
A: `guardrails.check_unverified_completion()` — every turn tracks whether a
*real* order was placed or a *real* email was sent (based on actual tool
results, not the model's text). If the model's reply contains a completion
claim ("order placed," "confirmation sent," etc. — matched via a
deliberately broad keyword list, not just literal past-tense verbs) without
a matching real event this turn, the reply is replaced with a canned,
correct message instead. This directly caught a genuinely serious bug: the
model once fabricated a plausible-looking address, called the *real*
`save_address` tool with it, and then claimed the order was placed — with no
real order behind the claim at all.

**Q: How do you stop internal tool-call mechanics leaking into the chat as text?**
A: `guardrails.leaked_tool_intent()` scans the model's reply for the literal
snake_case name of any of the 6 tools appearing anywhere in the text (e.g.
"I'll call decline_out_of_scope," or raw `{"name": "start_order", ...}` JSON
narrated instead of actually invoked). This is deliberately broad — any
occurrence at all is treated as a leak — because a narrower, phrase-gated
version ("I'll call the...") missed real leaks phrased differently (e.g. "by
calling start_order with the product_id..."). There's no legitimate reason a
user-facing reply should ever contain a tool's internal name, so the check
doesn't need to be clever about it.

**Q: What happens when a leak is detected — do you just apologize and give up?**
A: Depends on which tool leaked. A leaked `decline_out_of_scope` intent
converts cleanly to the real canned decline. A leaked `lookup_symptom`
intent is actually *recoverable* — the leaked JSON still contains the real
symptom the model meant to look up, so `recover_leaked_lookup()` performs
the real lookup itself and returns a properly formatted product
recommendation, instead of a dead-end "please rephrase" that would lose the
user's answer. Any other leak falls back to a generic re-ask.

**Q: How does clarifying-question logic work, and why is it deterministic rather than left to the model?**
A: `CLARIFYING_QUESTIONS` in `agent.py` defines symptoms ambiguous enough
that the wrong product recommendation is a real mismatch, not just
suboptimal — cough (dry vs. wet: a dry-cough suppressant and a wet-cough
expectorant work by opposite mechanisms, so the wrong one is actively
counterproductive, not just suboptimal), fever (child vs. adult: only one
of four fever products is pediatric), and cold (child vs. adult: only one
of six cold products is pediatric, and it's itself a dry-cough
formulation — a child's *wet* cough still has no product, which is why
that specific combination declines rather than recommending anything).
Both asking the question and resolving the answer are fully deterministic
Python, bypassing the LLM entirely for that turn.
Why: I originally left this to a soft prompt hint (`[Clarify: ...]`
injected into context) and the model simply ignored it — it fabricated an
answer to a question it never actually asked ("since your cough is bringing
up mucus" — the user never said that) and recommended the wrong product
anyway, directly contradicting its own fabricated premise.

**Q: How do you handle a qualifier that's already given upfront, e.g. "my baby has a fever"?**
A: The clarifying question correctly doesn't re-ask (the qualifier "baby" is
already present) — but the *first* version of this just fell through to an
unfiltered lookup, so a later bare "2" could still select an adult-only
product for a baby. Fixed by resolving pre-qualified messages through the
same branch-matching logic used for question answers, so "my baby has a
fever" goes straight to the one pediatric product deterministically.

**Q: How do you prevent the model from silently switching context/category mid-conversation?**
A: This was a real, serious bug: asking "last one" right after being shown a
list of adult *fever* products produced a completely unrelated *cold*
product list — the model called `lookup_symptom(symptom="cold")` with
nothing in the message suggesting cold at all. Root cause: the grounding
check (above) correctly allows a lookup call when "a category was
established earlier this session," but never verified the model's requested
category actually *matched* the established one. Fix: when a lookup is only
grounded by session history (not the current message's own text), the code
now force-corrects to the established category rather than trusting
whatever the model requested.

**Q: Are financially/practically consequential steps (placing an order, sending an email) ever fully automated end-to-end by the LLM?**
A: No, deliberately. `start_order` never completes synchronously — it always
defers to a follow-up question (ask for an address, or confirm the one on
file), and the *actual* order creation happens in a separate deterministic
function (`_complete_pending_order`/`_complete_confirmed_order` in
`main.py`) triggered by the user's next message, never by the model calling
a tool a second time to "confirm." This collapsed what used to be a
multi-tool chain (save_address → place_order → send_confirmation_email) that
the model couldn't reliably keep track of across turns into a single
deferred step with a deterministic completion.

**Q: Do you ever ask for confirmation before reusing a saved address?**
A: Always, explicitly — even when an address is already on file, `start_order`
asks the user to confirm it's still correct rather than silently reusing it
("people move — never assume a saved address is still correct"). This was
an explicit requirement, not just an engineering nicety.

**Q: Is there a limit on order quantity, and how is it enforced?**
A: `MAX_QUANTITY_PER_ORDER = 2` — a deliberately labeled temporary
placeholder (explicitly commented as such in `tools.py`), not sourced from
real regulatory guidance, since properly sourcing genuine per-medicine
quantity limits (they vary by drug, jurisdiction, and pack size) was out of
scope for the timeline. It's enforced server-side in `_clamp_quantity()`,
so the model cannot bypass it by requesting a higher quantity — and the
clamp is disclosed to the user *before* they confirm the order, not just
buried in the final confirmation (a real bug found and fixed: the model
received the clamp note in its tool result but silently omitted mentioning
it).

---

## 9. Hallucinations — Real Bugs Found and Fixed

*(This section is your strongest material if asked "what went wrong and how
did you fix it" — these are all real, reproduced, verified bugs, not
hypotheticals.)*

**Q: What's the most serious hallucination you found?**
A: The model fabricated a plausible-looking shipping address out of thin
air, called the *real* `save_address` tool with it, and then claimed the
order was placed — with no actual order behind the claim. Verified via the
dashboard endpoint that the address had been silently corrupted with no real
order to match. Fixed by removing the LLM from that specific state
transition entirely — once an address is pending, the *next* message is
either treated as an address deterministically or re-asked for, never
routed back through the model's own judgment.

**Q: Give me a concrete example of a category-switching hallucination.**
A: "Last one," asked right after a list of adult fever products, produced an
entirely unrelated cold product list — the model called
`lookup_symptom(symptom="cold")` despite nothing in the message suggesting
cold. Root cause and fix are in section 8 (grounding-category cross-check).
Two-layer fix: deterministic ordinal-phrase resolution ("last one," "the
third item bro") in `main.py` for the common case, plus a structural
category-match guard in `agent.py` as a backstop for any phrasing the
ordinal resolver doesn't catch.

**Q: Give me an example where the model contradicted its own reasoning.**
A: Asked to clarify a cough as dry vs. wet, the model — when the clarifying
question was only a soft prompt hint rather than a hard code path —
fabricated a premise ("since your cough is bringing up mucus") the user
never stated, then recommended the dry-cough product anyway, directly
contradicting the fabricated premise it had just invented. This is what
justified making clarifying questions fully deterministic rather than
prompt-guided.

**Q: Give me an example of a RAG-specific hallucination.**
A: Before `lookup_medicine_info` existed as a separate tool,
"side effects of cetirizine" was misrouted through `lookup_symptom`
(because "cetirizine" is also a cold keyword), which returned a bare
product list with no side-effect information at all — and the model, rather
than admitting it had nothing grounded to answer from, fabricated an entire
fake FDA-style answer complete with a bogus citation. Fixed by detecting
info-style questions via regex (`side effects|dosage|warning|interaction|...`)
and redirecting them to the correct RAG tool before the model ever gets a
chance to improvise.

**Q: How do you generally categorize the hallucination failure modes you found?**
A: Roughly four buckets: (1) **fabricated tool arguments** — inventing an
address, inventing a category to look up; (2) **unverified completion
claims** — describing an action as done with no real tool call behind it;
(3) **contradicted-premise answers** — inventing a fact mid-reasoning and
then acting on it; (4) **tool-name/mechanism leaks** — narrating internal
tool-calling machinery as user-facing text instead of using it. Each has a
dedicated, tested guardrail (section 8).

**Q: Would a bigger model (GPT-4-class) have avoided all of these?**
A: Probably fewer of them, but not zero, and that's somewhat beside the
point for this project's constraint (free/local only). More importantly:
even a much larger model benefits from structural guardrails for
consequential actions (placing real orders, claiming real completions) —
the "verify, don't just trust" philosophy isn't a workaround for a small
model specifically, it's good practice for any agent taking real-world
actions, just *more urgently necessary* at 3B scale.

---

## 10. Evaluation

**Q: How do you evaluate retrieval quality?**
A: `rag_eval.py` — a golden query set, 14 queries against
`retrieval.search()`, checked two ways. 10 in-scope queries (one per
catalog product) each carry a **golden reference answer** taken directly
from the corresponding `knowledge_base/*.md` file, plus a `must_include`
keyword checklist (e.g. for "dosage for paracetamol 500mg": `["4-6 hours",
"4000mg"]`). Each case is scored on (1) **source hit@k** — did the right
document come back at all — and (2) **content grounding** — are the actual
facts the reference answer depends on present across the top-k chunks
retrieval returns (what `lookup_medicine_info` really hands the model), not
just the right filename. 4 more queries are deliberately out-of-scope
(weather, coding help, a made-up drug, a refund policy) and must retrieve
*nothing* — those guard the `MIN_SIMILARITY` scope boundary and matter as
much as the positive cases. Every run also writes a self-contained HTML
report (`app/data/rag_eval_report.html`) — query, golden answer, and every
retrieved chunk with its score, side by side — so results are reviewable
at a glance instead of read off the terminal.

**Q: Why hit@k plus a keyword checklist, and not a full precision/recall/MRR suite or an LLM-judge similarity score?**
A: Proportionality again — at a 40-chunk corpus with 10 unambiguous
products, this is enough to catch real regressions (and it has, repeatedly
— see below) without building evaluation infrastructure disproportionate
to the corpus size, or introducing the cost/nondeterminism of an LLM judge
for grading. It's currently 14/14 (100%).

**Q: Did this evaluation suite actually catch real bugs, or is it just for show?**
A: Twice, concretely. First, it's what caught the `MIN_SIMILARITY`
threshold being miscalibrated early on (0.3 was letting an unrelated
"credit card refund policy" query return 3 false-positive results). Second,
adding the content-grounding check (beyond plain source hit@k) surfaced a
real, subtler gap the simpler version couldn't see: "dosage for cough
suppressant syrup" retrieved the right *document* (col-004.md), but its
actual Dosage section scored 0.606 — just under the 0.63 cutoff — losing to
its own Overview (0.672) and Warnings (0.654) sections, all three being
short, topically similar paragraphs about the same product with nothing in
the query pointing the embedding at Dosage specifically. I fixed this the
same way the existing `STRENGTH_BOOST` hybrid technique already worked
(section 6) rather than loosening the threshold globally: a second boost,
`SECTION_BOOST`, that only fires when the query names both a section intent
("dosage"/"side effects"/"warning") *and* the specific product (shares a
distinctive word with the chunk's own title) — gating on the product match
was load-bearing, since a first version without it boosted *every*
product's Dosage section indiscriminately, which undid the fix and turned
"dosage for amoxicillin" into a new false positive. I re-ran the full eval
suite after every change to `tools.py`, `agent.py`, `retrieval.py`, or any
threshold value throughout the project to confirm nothing regressed —
including right after migrating from the flat-JSON/manual-cosine
implementation to Chroma, and again after this SECTION_BOOST fix.

**Q: How do you evaluate the agent/conversation layer, as opposed to pure retrieval?**
A: There's no automated eval suite for the conversational layer (that's a
real gap, worth naming honestly if asked) — that side was validated through
live, manual, scenario-based testing: reproducing exact reported transcripts
(greetings, out-of-scope declines, multi-symptom messages, clarifying
questions, address-confirmation flows, ordinal product selection) via
scripted `curl` sequences and live browser testing after every change,
treating each fixed bug as a permanent regression case I re-ran on
subsequent changes. It's not automated, but it is systematic and repeated.

**Q: If you had more time, what would you add to evaluation?**
A: An automated conversation-level eval set — a fixed list of multi-turn
scripted transcripts (the same ones used for manual regression testing,
formalized) asserting on specific reply properties (contains "order
confirmed," doesn't contain a tool name, asks a specific clarifying
question) so the manual regression sweep becomes a `pytest` run. I'd also
add retrieval evaluation beyond hit@k — e.g. mean reciprocal rank, to
catch a *correct-but-lower-ranked* result becoming worse over time even
while still technically appearing in the top-3.

**Q: How do you know the guardrails themselves work, versus just believing your own code comments?**
A: Every guardrail fix in this project follows the same pattern: reproduce
the exact failing scenario first (via the real chat endpoint, not a unit
test in isolation), confirm the fix resolves it, then re-run the full
regression sweep (RAG eval + a battery of known-good conversational flows)
to confirm nothing else broke. That reproduce-fix-reverify loop is the
actual evaluation methodology for the agent/guardrail layer, even though
it isn't automated.

---

## 11. Persistence / Data Layer

**Q: Is any of this persisted, or does everything reset if the server restarts?**
A: It's genuinely persisted now (this was a real gap fixed later in the
project). `store.py` is backed by SQLite (`app/data/app.db`, created
automatically) — orders, the user's saved address/email, and all
in-progress session state (pending clarifications, pending address
confirmations, conversation history) survive a server restart. The RAG
index is likewise persisted in the Chroma collection on disk, not rebuilt
from scratch every startup (thanks to the content-hash cache check).

**Q: Why SQLite instead of Postgres/MySQL?**
A: Zero-setup, built into Python's standard library, a single file — exactly
proportionate to a single-demo-user local app with no concurrent-write
contention to speak of. Postgres would be the right call the moment there's
real multi-user concurrent traffic; that's not this system's shape.

**Q: How do you handle concurrent access to SQLite from multiple requests?**
A: A fresh short-lived connection per function call rather than one shared
connection — FastAPI's sync routes run in a thread pool, and sqlite3
connections aren't safe to share across threads. SQLite itself serializes
writes at the file level regardless, and this app's traffic profile (a
single live demo user) makes the per-call connection overhead a total
non-issue.

**Q: What's the schema?**
A: Three tables: `users` (user_id, name, address, email), `orders`
(order_id, user_id, product details, quantity, total, address), and
`session_state` (one row per conversation session — messages as JSON,
plus each "pending" field as its own column). Order ids are derived from
the highest existing numeric suffix already in the table rather than an
in-memory counter, specifically so a restart can never collide with or
reuse an id from before it.

**Q: Did adding real persistence surface any bugs that didn't exist before?**
A: Yes, one real one, and it's a good example of persistence exposing a
latent issue: Ollama's tool-call objects are typed Pydantic models, not
plain dicts. Storing them directly into in-memory conversation history was
harmless, but the moment `save_session_messages` needed to `json.dumps()`
that history for real SQLite storage, it crashed with `TypeError: Object of
type ToolCall is not JSON serializable` — meaning any conversation that
had made a tool call earlier would then fail on its *next* turn. Fixed by
converting tool calls to plain dicts via `.model_dump()` before storing.
Caught by testing the exact restart-survival scenario end-to-end, not by
inspection.

---

## 12. Hard / Adversarial Questions

**Q: This whole thing runs on a 3B model — isn't that a toy? Why should anyone trust it?**
A: For a genuinely open-ended assistant, yes, that skepticism is fair. But
this isn't an open-ended assistant — the design deliberately narrows what
the model is *allowed* to affect. It can recommend from a fixed 10-item
catalog, and it can trigger a small number of tools whose actual behavior
(what gets saved, what gets ordered, what claims are allowed through) is
enforced in plain Python, verified against real tool results, independent
of what the model says. The model supplies judgment and language; the code
supplies truth and safety.

**Q: What would break first if you scaled the catalog from 10 products to 800?**
A: Several things, roughly in this order: (1) `tools.classify_categories`'s
keyword/fuzzy matching, which is fine for 2 broad categories, would need to
become real retrieval-based product matching; (2) the knowledge base's
header-based chunking assumption (uniform template) would likely break
across more varied real-world sources, needing the hybrid chunking approach
discussed in section 3; (3) `MIN_SIMILARITY`/`STRENGTH_BOOST` would need
re-calibration against a much larger, more collision-prone embedding space;
(4) `_next_order_id`'s full-table scan is fine at demo scale but should
become a proper auto-increment or UUID at real scale.

**Q: How would you prevent prompt injection — e.g. a malicious image or message trying to manipulate the agent?**
A: The scope-limiting design already does a lot of this work structurally —
the model's only levers are 6 narrowly-scoped tools that can't act outside
the fixed catalog, so even a successfully "injected" instruction has almost
nothing dangerous to actually do. The image-analysis path is a real
attack surface worth naming honestly: a photo could contain adversarial text
aimed at the vision model. I haven't added an explicit
prompt-injection-detection layer for that path — a reasonable next step
would be treating Gemini's image description as untrusted input and
re-validating that any resulting tool call is still grounded in real catalog
data before acting on it (which the existing `symptom_lookup_grounded` cross
check already does, incidentally, since it doesn't special-case the image
path).

**Q: What's your latency like, and does the free/local approach cost you responsiveness?**
A: Yes, tangibly — local CPU inference on a 3B model plus up to 6 possible
tool-call round trips per turn is slower than a hosted frontier model API
would be. That's an accepted tradeoff for a zero-cost, fully local
architecture; a single retry (`_invoke_agent_with_retry`) also exists
specifically because a local Ollama instance under load occasionally has a
slow or failed first call.

**Q: If you had to pick one thing to redo differently from the start, what would it be?**
A: Building the deterministic state-machine layer (pending clarifications,
pending orders, pending address confirmations) from day one instead of
retrofitting it after discovering each failure mode live. In hindsight, the
pattern was predictable from the start — a small local model plus multi-turn
state always needed this — but I only converged on it by finding the
failures one at a time through live testing.

**Q: Why not just add a bigger/hosted model as a fallback for hard cases?**
A: That would break the explicit "free resources only" constraint set at
the start of this project, and — more importantly — it would mask rather
than fix the underlying reliability problem. A hosted fallback model doesn't
know when to trigger any better than the primary model does; the
deterministic guardrail approach fixes the actual failure mode regardless of
which model is generating text.
