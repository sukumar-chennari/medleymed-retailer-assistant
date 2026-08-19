# MedleyMed Retailer Assistant Demo

A small fever/cold OTC medicine assistant, styled as a prototype for MedleyMed's
retailer portal (navy/teal branding, "Retailer" terminology, matching
https://telemedicine.medleymed.com's login screen). Lands on a simple retailer
dashboard (shipping address, recent orders) with a floating chat bubble — click
it to open the assistant, describe symptoms or upload a photo of a medicine
label/prescription, get a product suggestion from a fixed catalog, and place a
demo order with email confirmation.

Scope is deliberately narrow: fever and cold only, one hardcoded demo user,
in-memory storage. Built entirely on free resources: **Ollama** (`llama3.2`,
running locally) drives the conversation and tool-calling, and the free
**Gemini API** reads uploaded photos. See
`/Users/macbookpro/.claude/plans/twinkly-humming-wolf.md` for the full design
plan and roadmap.

## Setup

1. Install and start [Ollama](https://ollama.com), then pull the text model:
   ```
   ollama pull llama3.2
   ```
   (Ollama must be running — `ollama serve`, or just have the app open.)
2. Get a free Gemini API key at https://aistudio.google.com/apikey (no credit
   card required) — this is only used to read uploaded photos.
3. `python -m venv .venv && source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. `cp .env.example .env` and fill in `GEMINI_API_KEY`. The `SMTP_*` vars are
   optional — without them, order confirmation emails are logged instead of sent.
6. `uvicorn app.main:app --reload`
7. Open http://localhost:8000

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
```

Every time you want to run it:

```bash
ollama serve            # if Ollama isn't already running
source .venv/bin/activate
uvicorn app.main:app --reload
```

Then open http://localhost:8000 in a browser.

To stop the server, press `Ctrl+C` in that terminal (or `pkill -f "uvicorn app.main:app"`).
