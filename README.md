# MedleyMed Retailer Assistant Demo

A small fever/cold OTC medicine assistant, styled as a prototype for MedleyMed's
retailer portal (navy/teal branding, "Retailer" terminology, matching
https://telemedicine.medleymed.com's login screen). Lands on a simple retailer
dashboard (shipping address, recent orders) with a floating chat bubble — click
it to open the assistant, describe symptoms or upload a photo of a medicine
label/prescription, get a product suggestion from a fixed catalog, and place a
demo order with email confirmation.

Scope is deliberately narrow: fever and cold only, one hardcoded demo user.
Built entirely on free resources: **Ollama** (`llama3.2`, running locally)
drives the conversation and tool-calling, `nomic-embed-text` (also local
Ollama) powers a small retrieval-augmented-generation (RAG) knowledge base
for medicine dosage/side-effect questions, and the free **Gemini API** reads
uploaded photos. See `DEMO_QA_PREP.md` for the design rationale and demo Q&A.

App data (orders, addresses, in-progress conversation state) persists in a
local SQLite file and survives restarts; the RAG knowledge base is embedded
into a local Chroma vector store rather than plain in-memory Python — both
run entirely on-disk with no server or paid service involved.

### Code layout

- `app/agent.py` — the agent: system prompt, tool schema, tool-calling loop
- `app/guardrails.py` — hallucination/leak/pleasantry safety-net checks
- `app/tools.py` — tool implementations (catalog lookup, orders, RAG lookup)
- `app/data_ingest.py` — chunks, embeds, and upserts the knowledge base into
  Chroma (run standalone with `python -m app.data_ingest`)
- `app/retrieval.py` — vector search over the Chroma collection
- `app/data/knowledge_base/*.md` — the RAG corpus (one file per catalog item)
- `app/store.py` — SQLite-backed app state (orders, sessions, addresses) —
  `app/data/app.db`, created automatically on first run
- `app/main.py` — FastAPI routes and the deterministic order/address/email
  conversation flow
- `app/schemas.py` — request/response models, including the input limits
- `app/config.py` — environment-variable settings (see `.env.example`)
- `app/rag_eval.py`, `app/conversation_eval.py` — standalone eval scripts
  (see Tests below), not part of the running app
- `static/` — the dashboard and chat widget (plain HTML/CSS/JS, no build step)
- `tests/` — the pytest suite; `.github/workflows/tests.yml` runs it in CI

## Setup

1. Install and start [Ollama](https://ollama.com), then pull the models:
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
   (Ollama must be running — `ollama serve`, or just have the app open.)
2. Get a free Gemini API key at https://aistudio.google.com/apikey (no credit
   card required) — this is only used to read uploaded photos.
3. `python -m venv .venv && source .venv/bin/activate` — needs Python 3.10 or
   newer (the code uses `str | None` type syntax); CI runs 3.11
4. `pip install -r requirements.txt`
5. `cp .env.example .env` and fill in `GEMINI_API_KEY`. The `SMTP_*` vars are
   optional — without them, order confirmation emails are logged instead of sent.
6. `uvicorn app.main:app --reload`
7. Open http://localhost:8000

The first run automatically builds the RAG index (a local Chroma collection
at `app/data/chroma_db/`) if it doesn't exist yet. To rebuild it explicitly
(e.g. after editing the knowledge base), run `python -m app.data_ingest`.

App data lives in `app/data/app.db` (SQLite), created automatically on first
run. To reset the demo to a clean slate (no orders, no saved address/email, no
chat sessions or metrics), click **Reset Demo Data** in the dashboard's top bar
(or `POST /api/reset-demo`) — it works while the server is running. You can
also delete the file instead, but stop the server first: the tables are only
created at startup, so deleting `app.db` under a running server makes every
request fail with `no such table` until you restart it.

## Running the demo

Everything runs on your own machine — no cloud deployment needed. Just make sure
Ollama is running and this server is up before the demo, on the same laptop the
browser is opened on.

## Commands

One-time setup (see Setup above for details):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
ollama pull llama3.2
ollama pull nomic-embed-text
```

Every time you want to run it:

```bash
ollama serve            # if Ollama isn't already running
source .venv/bin/activate
uvicorn app.main:app --reload
```

Then open http://localhost:8000 in a browser.

To stop the server, press `Ctrl+C` in that terminal (or `pkill -f "uvicorn app.main:app"`).

## Tests

`tests/` covers every deterministic part of the app — one or more files per
module (`test_guardrails.py`, `test_tools*.py`, `test_data_ingest.py`,
`test_agent*.py`, `test_main*.py`, `test_retrieval*.py`, `test_store.py`),
plus `test_catalog_integrity.py` and `test_frontend_schema_sync.py` (which
keeps the hand-copied limits in `static/` in step with `app/schemas.py`).
Anything touching the database runs against an isolated temp DB (see
`tests/conftest.py`'s `isolated_db` fixture), never the real
`app/data/app.db` the live demo uses. Needs no *chat* model — but
`test_retrieval.py`/`test_retrieval_search.py` do call the real, local
embedding model (`nomic-embed-text`) against the real, already-ingested
knowledge base, since unlike the chat model that call has no sampling
anywhere in it (verified: the same query returns byte-identical results
every time), so it's safe to assert on exactly — not flaky the way an LLM
*generation* would be. Runs in a couple seconds:

```bash
python -m pytest tests/ -v
```

To see coverage (which lines are actually exercised, module by module):

```bash
python -m pytest --cov=app --cov-report=term-missing tests/
```

`.coveragerc` excludes `app/rag_eval.py` and `app/conversation_eval.py` from
that report — they're standalone scripts run directly (`python -m
app.rag_eval`, `python -m app.conversation_eval`), not application code these
tests exercise; their own methodology is the golden-query and scripted-
conversation evals below, not something meant to have unit tests. The one
thing genuinely needing a live LLM call and left uncovered here is
`run_turn`'s LLM-invoking body; the LangChain tool wrappers and
`_GuardrailMiddleware` don't need a model at all and are covered
(`test_agent_build_tools.py`, `test_agent_guardrail_middleware.py`). The live
part stays in the manual/live-testing category described below — an LLM reply
is nondeterministic enough that asserting on exact text would be flaky.

This doesn't replace `python -m app.rag_eval`, which needs the real agent
and knowledge base and checks a different thing (retrieval/answer quality
against golden queries, not guardrail correctness) — or
`python -m app.conversation_eval`, which drives scripted multi-turn
conversations through the real `/api/chat` route (the actual pipeline,
`main.py`'s pending-state dispatch chain together with the LLM's own
tool-calling) and asserts on structural reply properties (contains "Order
confirmed!" with a real order id, cites a real source, doesn't leak a
tool name) robust to the model's own wording, rather than exact text.
Runs against an isolated temp database, never the real demo data. Both of
these are intentionally slow and excluded from CI, the same reasoning as
above.

`.github/workflows/tests.yml` runs this same suite on every push/PR to
`main` — it installs Ollama and pulls `nomic-embed-text` first, since
importing `app.tools` pulls in `app.retrieval`, which builds/loads the RAG
index at import time.
