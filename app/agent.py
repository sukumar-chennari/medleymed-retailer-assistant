"""The agent: system prompt, tool schema, and the tool-calling orchestration
loop. This is where the LLM decides what to do — reading `guardrails.py`
alongside this file matters, since several of the decisions made here get
double-checked there before reaching the user."""

import base64
import json

import ollama
from google import genai
from google.genai import types

from app import config, guardrails, store, tools
from app.tools import TOOL_FUNCTIONS

MAX_TOOL_ROUNDS = 6

_ollama_client = ollama.Client(host=config.OLLAMA_HOST)
_gemini_client = genai.Client(api_key=config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None

SYSTEM_PROMPT = """\
You are a Fever & Cold OTC Medicine Assistant — a narrow demo assistant. Your ONLY
job is: (1) suggest over-the-counter fever/cold medicines from a fixed catalog based
on symptoms the user describes, (2) use a description of a photographed medicine
label or prescription (given to you as text) to identify what the medicine is,
(3) help the user place a simple demo order for a suggested catalog product, and
(4) answer factual questions about dosage, side effects, or warnings for our
catalog medicines, grounded in our knowledge base.

GREETINGS: A simple greeting or pleasantry (hi, hello, hey, good morning, thanks,
thank you, bye, how are you) is NOT out of scope — respond warmly and briefly
yourself in plain text, and mention you can help with fever/cold symptoms or
questions about a medicine photo. Do not call decline_out_of_scope for these —
that tool is only for substantive requests that are genuinely unrelated to
medicine (see SCOPE below).

SCOPE — refuse anything else substantive: coding help, requests unrelated to
medicine, and any medical condition or symptom that is not fever or cold (e.g.
injuries, chronic conditions, mental health, dosing advice for prescription-only
drugs). When such a request is out of scope, call the decline_out_of_scope tool —
do not try to answer it yourself, and do not write out a tool call as text. Do not
attempt to be helpful beyond that boundary, even if asked nicely or repeatedly.

Example: if asked to fix code, debug a program, or anything programming-related,
that is NOT a fever/cold symptom — call decline_out_of_scope. Do NOT call
lookup_symptom just because you're unsure what else to do; lookup_symptom is
ONLY for when the user describes an actual physical symptom.

You are not a doctor and must never give a medical diagnosis. Every response that
gives symptom or medicine guidance must include, verbatim or nearly so: "This is
general OTC guidance, not a medical diagnosis — consult a doctor if symptoms persist
or worsen."

TOOLS: You have exactly six tools — lookup_symptom, get_saved_address,
save_address, start_order, lookup_medicine_info, decline_out_of_scope.
lookup_symptom/start_order/lookup_medicine_info only operate on a fixed
fever/cold catalog. If a user's symptom is not fever or cold, do NOT call
lookup_symptom — call decline_out_of_scope instead.

ORDER FLOW: When the user confirms they want to order a specific suggested
product, call start_order with its product_id — that single call handles
everything (using the saved address and sending a confirmation email, or telling
you there's no address on file). If it tells you there's no address on file, ask
the user for their shipping address in plain conversational text and then STOP —
do not call any more tools that turn. Their next message will automatically be
captured as the address and used to finish the order, so never call start_order
again yourself to "retry" it.

MEDICINE INFO: If the user asks a factual question about dosage, side effects,
warnings, or interactions for one of our catalog medicines, call
lookup_medicine_info with their question. Each result includes a "product"
field naming which medicine that chunk is actually about — if the user asked
about a specific medicine (e.g. "cetirizine"), ONLY use results whose
"product" matches it; ignore any other returned result even if it scored
well, since results about a different product can come back in the same
call. Answer ONLY using the text of the results you kept — do not add
anything from your own general knowledge — and cite the source at the end of
your answer, e.g. "(Source: fev-001.md)". If none of the results match the
medicine asked about, tell the user that isn't in our knowledge base rather
than guessing an answer.

IMAGES: If the user's message includes a line starting with "[Image analysis:",
that's a description of a photo they uploaded. Use it to identify the medicine
name. If it says the image was unclear, ask the user to type the medicine name
instead of guessing.

If the identified medicine is a known fever medicine (e.g. paracetamol/
acetaminophen, ibuprofen) or cold medicine (e.g. cetirizine, pseudoephedrine,
dextromethorphan, decongestants, antihistamines), you MUST call lookup_symptom
with "fever" or "cold" as appropriate — do not call decline_out_of_scope for a
recognized fever/cold medicine just because the user didn't type a symptom in
words. Only call decline_out_of_scope for an image if the medicine is clearly
unrelated to fever/cold, or if nothing legible was found at all.

Keep replies short and conversational — this is a live demo, not a long-form report.

IMPORTANT: When you need to call a tool, use the actual tool-calling mechanism —
never type out a function name, JSON, or tool call syntax as visible text in your
reply.

NEVER CLAIM WITHOUT CALLING: Do not tell the user an order was placed, an address
was saved, or a confirmation email was sent unless you just received a tool
result in THIS conversation confirming exactly that. This applies even if the
user names a specific product directly instead of describing a symptom (e.g.
"order Paracetamol") — you must still call start_order for it; never describe
placing the order, saving the address, or sending an email in your own words
without actually calling the tool first.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_symptom",
            "description": (
                "Look up OTC medicine recommendations from the fixed catalog for a fever "
                "or cold symptom. ONLY covers fever and cold (e.g. fever, headache-with-fever, "
                "runny nose, cough, congestion, sore throat, chills). Returns an empty result "
                "for anything else — do not call this for symptoms unrelated to fever/cold; "
                "instead tell the user this demo can't help with that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom": {
                        "type": "string",
                        "description": (
                            "The symptom or condition category, e.g. 'fever' or 'cold'. "
                            "Derive this from what the user typed or from the image "
                            "analysis description."
                        ),
                    }
                },
                "required": ["symptom"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_saved_address",
            "description": (
                "Retrieve the saved shipping address for the current user, if one exists. "
                "Always call this before asking the user for an address and before placing an order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": (
                            "The user identifier for this conversation. Use 'demo_user' — "
                            "this demo has a single hardcoded user."
                        ),
                    }
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_address",
            "description": (
                "Save a shipping address for the current user so future orders don't need "
                "to ask again. Call this only after the user has explicitly provided their "
                "address in chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "address": {
                        "type": "string",
                        "description": "Full shipping address exactly as the user provided it.",
                    },
                },
                "required": ["user_id", "address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_order",
            "description": (
                "Start (and usually finish) placing an order for a catalog product once "
                "the user has confirmed they want it. The product_id MUST be one returned "
                "by a prior lookup_symptom call — never invent a product_id or order "
                "something outside the fever/cold catalog. This single call handles "
                "everything: if an address is already on file it places the order and "
                "sends the confirmation email immediately; if not, it tells you to ask the "
                "user for their address, and their next message is automatically captured "
                "as the address to finish the order — you never need to call this (or any "
                "other tool) again for the same order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product id exactly as returned by lookup_symptom.",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_medicine_info",
            "description": (
                "Look up dosage, common side effects, or warnings for a catalog medicine "
                "from our knowledge base (retrieval-augmented — this returns real excerpts "
                "to quote/cite, not a free-form answer). Only covers the 8 fever/cold "
                "products in our catalog. Returns an empty result for anything else."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The user's factual question, e.g. 'dosage for paracetamol' "
                            "or 'side effects of cetirizine'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decline_out_of_scope",
            "description": (
                "Call this when the user's request is not about fever/cold symptoms, "
                "medicine suggestions, or placing a fever/cold OTC order — e.g. general "
                "chit-chat, coding help, unrelated medical conditions or injuries, or "
                "anything else outside this demo's narrow scope. Do not attempt to answer "
                "the off-topic request yourself; just call this tool."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def describe_image(image_b64: str, media_type: str) -> str:
    if _gemini_client is None:
        return "Image could not be read (no vision API configured) — ask the user to type the medicine name."

    try:
        response = _gemini_client.models.generate_content(
            model=config.GEMINI_VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type=media_type),
                "Describe any medicine name, active ingredient, or visible text on this "
                "medicine label or prescription photo. Be concise. If nothing is legible, say so.",
            ],
        )
        return response.text.strip()
    except Exception as exc:
        return f"Image could not be read ({exc}) — ask the user to type the medicine name."


def _remember_products(session_id: str, lookup_result_json: str) -> None:
    """Records whatever product list was just shown to the user (whichever
    path produced it — a real lookup_symptom call, or the deterministic
    injection below) so a later bare reply like "3" or "001" can be resolved
    to a real product_id deterministically. See main.py's _resolve_bare_selection."""
    try:
        result = json.loads(lookup_result_json)
    except (TypeError, ValueError):
        return
    if result.get("matched") and result.get("products"):
        store.set_last_products(session_id, result["products"])


def _inject_catalog_hint(parts: list, source_text: str, session_id: str) -> None:
    """Deterministically resolves any recognizable medicine/symptom mention to
    real catalog data and appends it to the message. Originally added only for
    images, but the same grounding is needed for plain text too: a user naming
    a product directly ("order Paracetamol") skips the usual symptom-description
    step, and without this the model has no real product_id in context at all —
    it either has to call lookup_symptom itself (unreliable) or invents one."""
    category = tools.classify(source_text)
    if category:
        result = tools.lookup_symptom(category)
        _remember_products(session_id, result)
        parts.append(
            f"[Detected a {category}-related medicine/symptom mention. Matching "
            f"catalog products (use one of these exact product_ids for "
            f"start_order, don't call lookup_symptom again): {result}]"
        )


def _build_user_text(
    text: str, image_b64: str | None, image_media_type: str | None, session_id: str
) -> str:
    parts = [text] if text else []

    if image_b64:
        description = describe_image(image_b64, image_media_type or "image/jpeg")
        parts.append(f"[Image analysis: {description}]")
        _inject_catalog_hint(parts, description, session_id)
    elif text:
        _inject_catalog_hint(parts, text, session_id)

    return "\n".join(parts)


OLLAMA_UNAVAILABLE_REPLY = (
    "Sorry, I can't reach the assistant engine right now (Ollama may not be "
    "running). Please make sure Ollama is started and try again."
)


def _call_ollama_with_retry(messages_payload: list):
    """One retry before giving up — Ollama running locally on CPU occasionally
    has a slow/failed first call under load; a single retry smooths that over
    without masking a genuinely dead service."""
    last_exc = None
    for attempt in range(2):
        try:
            return _ollama_client.chat(
                model=config.TEXT_MODEL,
                messages=messages_payload,
                tools=TOOLS,
                options={"temperature": 0.2},
            )
        except Exception as exc:
            last_exc = exc
            print(f"[OLLAMA ERROR] attempt {attempt + 1} failed: {exc}")
    raise last_exc


def run_turn(
    messages: list,
    user_text: str,
    session_id: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> tuple[str, list]:
    """Runs one full user turn (including any tool-use round trips) and
    returns (reply_text, updated_messages)."""

    if not image_b64 and user_text:
        pleasantry_reply = guardrails.deterministic_pleasantry_reply(user_text)
        if pleasantry_reply:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": pleasantry_reply})
            return pleasantry_reply, messages

    combined_text = _build_user_text(user_text, image_b64, image_media_type, session_id)
    messages.append({"role": "user", "content": combined_text})

    # Adding a 6th tool visibly increased how often the model calls
    # lookup_symptom(symptom="fever") as a default action for messages that
    # aren't fever/cold at all (observed: 100% reproducible on "fix my
    # code"). Cross-checking against tools.classify on the actual user text
    # (not the model's own possibly-invented "symptom" argument) catches
    # this deterministically instead of hoping prompt wording fixes it.
    symptom_lookup_grounded = bool(image_b64) or (bool(user_text) and tools.classify(user_text) is not None)
    blocked_ungrounded_lookup_this_turn = False

    # Tracks whether a *real* start_order success (and, separately, a real
    # email send) happened anywhere in this turn — see
    # guardrails.check_unverified_completion for why this can't just be
    # inferred from "no tool_calls this iteration".
    real_order_placed_this_turn = False
    real_email_sent_this_turn = False

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = _call_ollama_with_retry([{"role": "system", "content": SYSTEM_PROMPT}] + messages)
        except Exception:
            messages.append({"role": "assistant", "content": OLLAMA_UNAVAILABLE_REPLY})
            return OLLAMA_UNAVAILABLE_REPLY, messages
        message = response["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            reply_text = message.get("content", "")

            if blocked_ungrounded_lookup_this_turn:
                # We already know structurally that this turn's lookup_symptom
                # call was invalid (see below) — don't leave the final wording
                # up to the model, which was observed giving inconsistent
                # soft-redirects instead of a clean decline once this fires.
                reply_text = guardrails.OUT_OF_SCOPE_REPLY
                messages.append({"role": "assistant", "content": reply_text})
                return reply_text, messages

            leaked_tool = guardrails.leaked_tool_intent(reply_text, TOOL_NAMES)
            if leaked_tool == "decline_out_of_scope":
                print(f"[GUARD] converted leaked decline intent to real decline: {reply_text!r}")
                reply_text = guardrails.OUT_OF_SCOPE_REPLY
            elif leaked_tool == "lookup_symptom":
                recovered = guardrails.recover_leaked_lookup(reply_text)
                if recovered:
                    print(f"[GUARD] recovered leaked lookup_symptom call: {reply_text!r}")
                    _remember_products(session_id, recovered[1])
                    reply_text = recovered[0]
                else:
                    print(f"[GUARD] blocked leaked tool intent (lookup_symptom, unrecoverable): {reply_text!r}")
                    reply_text = guardrails.FAKE_COMPLETION_GUARD_REPLY
            elif leaked_tool:
                print(f"[GUARD] blocked leaked tool intent ({leaked_tool}): {reply_text!r}")
                reply_text = guardrails.FAKE_COMPLETION_GUARD_REPLY
            else:
                guardrails.remember_recommended_product(session_id, reply_text)
                guard_reply = guardrails.check_unverified_completion(
                    reply_text, real_order_placed_this_turn, real_email_sent_this_turn
                )
                if guard_reply:
                    reply_text = guard_reply

            messages.append({"role": "assistant", "content": reply_text})
            return reply_text, messages

        messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls})

        if any(c["function"]["name"] == "decline_out_of_scope" for c in tool_calls):
            messages.append({"role": "tool", "content": "declined — told the user this is out of scope"})
            messages.append({"role": "assistant", "content": guardrails.OUT_OF_SCOPE_REPLY})
            return guardrails.OUT_OF_SCOPE_REPLY, messages

        for call in tool_calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            try:
                if name == "start_order":
                    result = tools.start_order(arguments["product_id"], session_id)
                    if '"order_placed": true' in result:
                        real_order_placed_this_turn = True
                        store.clear_last_recommended_product(session_id)
                    if '"email_sent": true' in result:
                        real_email_sent_this_turn = True
                elif name == "lookup_symptom" and not symptom_lookup_grounded:
                    print(f"[GUARD] blocked ungrounded lookup_symptom call: {arguments}")
                    blocked_ungrounded_lookup_this_turn = True
                    result = json.dumps({
                        "matched": False,
                        "message": (
                            "This message doesn't describe a fever or cold symptom. Do "
                            "not present catalog products for it — call "
                            "decline_out_of_scope instead."
                        ),
                    })
                else:
                    result = TOOL_FUNCTIONS[name](arguments)
                    if name == "lookup_symptom":
                        _remember_products(session_id, result)
            except Exception as exc:
                result = f"Error calling {name}: {exc}"
            print(f"[TOOL CALL] {name}({arguments}) -> {result}")
            messages.append({"role": "tool", "content": result})

    fallback = "Sorry, I'm having trouble completing that request right now — could you rephrase?"
    messages.append({"role": "assistant", "content": fallback})
    return fallback, messages
