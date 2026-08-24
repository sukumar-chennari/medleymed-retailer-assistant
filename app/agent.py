"""The agent: system prompt, tool schema, and the tool-calling orchestration
loop. This is where the LLM decides what to do — reading `guardrails.py`
alongside this file matters, since several of the decisions made here get
double-checked there before reaching the user."""

import base64
import json
import re

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

MULTIPLE SYMPTOMS: A user can describe more than one symptom at once (e.g.
"nose block and high temperature" — congestion AND fever). If the
lookup_symptom result's "category" field contains more than one category
(shown as "fever+cold"), you MUST recommend at least one product for EACH
category mentioned, not just whichever one you noticed first — the results
already include products from every matched category for exactly this
reason.

CLARIFYING QUESTIONS: When a symptom is ambiguous enough that the wrong
product could be genuinely unsuitable, ask a short narrowing question BEFORE
recommending a product, instead of guessing. If the user's message includes a
line starting with "[Clarify:", ask exactly that question instead of calling
lookup_symptom yet — wait for their answer first.

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
anything from your own general knowledge — and cite the source and section at
the end of your answer using both the "source" and "section" fields, e.g.
"(Source: fev-001.md § Dosage)". If you used more than one result, cite each
one. If none of the results match the medicine asked about, tell the user
that isn't in our knowledge base rather
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

# Catches an observed tool-misrouting failure: "side effects of cetirizine"
# was called through lookup_symptom instead of lookup_medicine_info, because
# "cetirizine" itself is a recognized cold keyword — lookup_symptom happily
# returned a generic cold-product list (no side-effect info at all), and the
# model then fabricated an entire fake FDA-style answer with a bogus citation
# rather than admitting it had nothing grounded to answer from. Any message
# matching this pattern is an info QUESTION about a medicine, never a symptom
# description, so a lookup_symptom call for it gets redirected to the correct
# tool instead of trusting the model to pick the right one every time.
INFO_QUESTION_RE = re.compile(
    r"\b(side\s*effects?|dosage|warning|interaction|overdose|how (much|many)\b.*\bmg)\b",
    re.IGNORECASE,
)


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


# Symptoms ambiguous enough that recommending the wrong product would be a
# real mismatch, not just a suboptimal pick — e.g. our only cough product
# (col-004) is a dry-cough suppressant, explicitly unsuited for a productive
# ("wet"/"chesty") cough per its own knowledge-base entry. Extensible: add
# more entries here for other symptoms worth narrowing down before
# recommending, following the same {qualifiers, question} shape.
CLARIFYING_QUESTIONS = {
    "cough": {
        "qualifiers": ("dry", "wet", "chesty", "productive", "phlegm", "mucus", "sputum"),
        "question": (
            "Is your cough dry, or is it bringing up mucus (a chesty/productive "
            "cough)? Our cough product is a dry-cough suppressant only, so this "
            "makes sure it's actually the right fit."
        ),
        "matching_answers": ("dry",),
        "non_matching_answers": ("wet", "chesty", "productive", "phlegm", "mucus", "sputum"),
        "matching_product_id": "col-004",
        "no_match_reply": (
            "Our only cough product (Cough Suppressant Syrup) is a dry-cough "
            "suppressant, so it isn't the right fit for a productive/chesty "
            "cough — I'd recommend checking with a pharmacist for that instead. "
            "Is there anything else I can help with?"
        ),
    },
}


def _needs_clarification(source_text: str) -> tuple[str, str] | None:
    """Returns (trigger, question) if source_text mentions an ambiguous
    symptom without its qualifier already present, else None."""
    s = source_text.lower()
    for trigger, rule in CLARIFYING_QUESTIONS.items():
        if trigger in s and not any(q in s for q in rule["qualifiers"]):
            return trigger, rule["question"]
    return None


def resolve_clarification(trigger: str, answer_text: str, session_id: str) -> str | None:
    """Deterministically resolves the user's answer to a previously-asked
    clarifying question — same reasoning as asking it deterministically:
    the model fabricated an answer to a question it never asked once
    (see run_turn), so the resolution doesn't get left to it either.
    Returns None if the answer doesn't clearly match either side, letting
    the caller fall through to a normal LLM turn instead of guessing."""
    rule = CLARIFYING_QUESTIONS.get(trigger)
    if not rule:
        return None
    s = answer_text.lower()

    if any(a in s for a in rule["matching_answers"]):
        product = store.find_product(rule["matching_product_id"])
        store.set_last_recommended_product(session_id, product["id"])
        return (
            f"{product['name']} would be a good fit — {product['description']}\n\n"
            f"Would you like to order this?\n\n{guardrails.DISCLAIMER}"
        )
    if any(a in s for a in rule["non_matching_answers"]):
        return rule["no_match_reply"]
    return None


def _inject_catalog_hint(parts: list, source_text: str, session_id: str) -> None:
    """Deterministically resolves any recognizable medicine/symptom mention to
    real catalog data and appends it to the message. Originally added only for
    images, but the same grounding is needed for plain text too: a user naming
    a product directly ("order Paracetamol") skips the usual symptom-description
    step, and without this the model has no real product_id in context at all —
    it either has to call lookup_symptom itself (unreliable) or invents one."""
    clarification = _needs_clarification(source_text)
    if clarification:
        # Deliberately skip the catalog-data injection below this turn — if
        # the model sees the actual product alongside the question, it's
        # liable to just recommend it anyway instead of asking first (the
        # same "won't admit it needs more info" pattern seen elsewhere). This
        # path only runs for the image-description flow now — plain text hits
        # the fully deterministic check in run_turn first, which this can't
        # override since it never reaches _build_user_text in that case.
        parts.append(f"[Clarify: {clarification[1]}]")
        return

    categories = tools.classify_categories(source_text)
    if categories:
        # Pass the original text, not a pre-classified single category — a
        # message can describe both ("nose block and fever"), and collapsing
        # to one category here before calling lookup_symptom would silently
        # drop the other one's products even though lookup_symptom itself
        # now handles multi-category text correctly.
        result = tools.lookup_symptom(source_text)
        _remember_products(session_id, result)
        category_label = "+".join(categories)
        parts.append(
            f"[Detected a {category_label}-related medicine/symptom mention. Matching "
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

        # Observed failure: the injected "[Clarify: ...]" hint (soft prompt
        # guidance) was ignored outright — the model fabricated an answer to
        # a question it never actually asked ("since your cough is bringing
        # up mucus" — the user never said that) and then recommended the
        # dry-cough product anyway, contradicting its own fabricated premise.
        # Asking the question is now deterministic and bypasses the LLM
        # entirely for this turn, the same way pleasantries do above —
        # there's no chance to skip or hallucinate around it.
        clarification = _needs_clarification(user_text)
        if clarification:
            trigger, question = clarification
            store.set_pending_clarification(session_id, trigger)
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": question})
            return question, messages

    combined_text = _build_user_text(user_text, image_b64, image_media_type, session_id)
    messages.append({"role": "user", "content": combined_text})

    # Adding a 6th tool visibly increased how often the model calls
    # lookup_symptom(symptom="fever") as a default action for messages that
    # aren't fever/cold at all (observed: 100% reproducible on "fix my
    # code"). Cross-checking against tools.classify on the actual user text
    # (not the model's own possibly-invented "symptom" argument) catches
    # this deterministically instead of hoping prompt wording fixes it.
    #
    # Checking classify() on the current message ALONE is too narrow though —
    # a natural follow-up like "any alternative?" doesn't repeat a symptom
    # keyword but is clearly still in-scope once a category was already
    # established (e.g. "runny nose" earlier this session). store.get_last_products
    # is the same "a category is already active in this conversation" signal
    # main.py's bare-selection handling relies on, so a lookup here is grounded
    # too even though this message's own text doesn't classify.
    symptom_lookup_grounded = (
        bool(image_b64)
        or (bool(user_text) and tools.classify(user_text) is not None)
        or bool(store.get_last_products(session_id))
    )
    blocked_ungrounded_lookup_this_turn = False

    # Tracks whether a *real* start_order success (and, separately, a real
    # email send) happened anywhere in this turn — see
    # guardrails.check_unverified_completion for why this can't just be
    # inferred from "no tool_calls this iteration".
    real_order_placed_this_turn = False
    real_email_sent_this_turn = False
    completed_order_this_turn = None

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

            if completed_order_this_turn:
                # A real order was placed this turn — render the confirmation
                # from the actual order data instead of the model's own
                # phrasing. Observed failure: the model sometimes free-generates
                # a vague confirmation ("please allow 3-5 business days...")
                # that omits the order id, price, and address the user needs,
                # even though the order itself was genuinely placed.
                reply_text = guardrails.build_order_confirmation(completed_order_this_turn)
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
            if symptom_lookup_grounded:
                # Mirrors the ungrounded-lookup_symptom guard, in reverse: the
                # model declined a message that's actually grounded (either
                # the text itself classifies, or a category is already
                # established this session) — e.g. "anything apart from
                # paracetamol?" got declined outright despite fever context
                # from the same turn. Redirect to a real lookup instead of
                # trusting the decline, rather than ending the turn wrong.
                category = tools.classify(user_text) if user_text else None
                if not category:
                    last = store.get_last_products(session_id)
                    if last:
                        category = "fever" if last[0]["id"].startswith("fev") else "cold"
                if category:
                    print(f"[GUARD] blocked incorrect decline_out_of_scope on grounded input: {user_text!r}")
                    result = tools.lookup_symptom(category)
                    _remember_products(session_id, result)
                    messages.append({"role": "tool", "content": result})
                    continue

            messages.append({"role": "tool", "content": "declined — told the user this is out of scope"})
            messages.append({"role": "assistant", "content": guardrails.OUT_OF_SCOPE_REPLY})
            return guardrails.OUT_OF_SCOPE_REPLY, messages

        for call in tool_calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            try:
                if name == "start_order":
                    result = tools.start_order(arguments["product_id"], session_id)
                    parsed_order = json.loads(result)
                    if parsed_order.get("order_placed"):
                        real_order_placed_this_turn = True
                        completed_order_this_turn = parsed_order
                        store.clear_last_recommended_product(session_id)
                    if parsed_order.get("email_sent"):
                        real_email_sent_this_turn = True
                elif name == "lookup_symptom" and user_text and INFO_QUESTION_RE.search(user_text):
                    print(f"[GUARD] redirected misrouted lookup_symptom to lookup_medicine_info: {user_text!r}")
                    result = tools.lookup_medicine_info(user_text)
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
