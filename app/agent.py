"""The agent: system prompt, tool schema, and the tool-calling orchestration
loop — built on LangChain's `create_agent` (LangGraph under the hood) rather
than a hand-rolled loop over the raw Ollama chat API. Reading `guardrails.py`
alongside this file matters, since several of the decisions made here get
double-checked there before reaching the user.

Every guardrail that used to live inline in a manual tool-calling loop is
re-implemented here as agent middleware: `wrap_tool_call` intercepts a tool
call *before* it executes (block it, redirect it to a different real action,
or correct its arguments), and `after_agent` overrides the final reply
*after* the model produces it — the same two intervention points the old
loop used, just expressed through LangChain's hook API instead of an
in-place `for` loop.
"""

import base64
import json
import re

from google import genai
from google.genai import types
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError

from app import config, guardrails, store, tools

MAX_TOOL_ROUNDS = 6

TOOL_NAMES = {
    "lookup_symptom", "get_saved_address", "save_address",
    "start_order", "lookup_medicine_info", "decline_out_of_scope",
}

_chat_model = ChatOllama(model=config.TEXT_MODEL, base_url=config.OLLAMA_HOST, temperature=0.2)
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
product, call start_order with its product_id (and quantity, if they mentioned
one). It never completes the order right away — it always defers to one of two
questions, which you relay to the user in plain conversational text and then
STOP, without calling any more tools that turn:
- No address on file: ask for their shipping address.
- An address IS on file: ask them to confirm it's still correct (quote the
  address back to them) — never assume a saved address hasn't changed.
Their next message automatically resolves whichever question this asked, so
never call start_order again yourself to "retry" or "confirm" it — that
happens deterministically, not through another tool call.

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

# Catches an observed failure where merely describing a symptom for the
# first time ("i have fever") produced an immediate start_order call and
# skipped ever asking "would you like to order this?" — the catalog hint
# injected below literally tells the model which product_id to use "for
# start_order", which the model took as license to call it right away. Since
# start_order always defers to an address ask/confirmation rather than
# completing silently, that turned a plain symptom mention into a confusing
# "should I ship here?" prompt for an order the user never agreed to. Only
# messages containing explicit ordering language bypass the guard below.
ORDER_INTENT_RE = re.compile(
    r"\b(order|buy|purchase|checkout|add to cart|get me|place (an|the) order)\b",
    re.IGNORECASE,
)

# Gates the decline_out_of_scope reversal's established-category fallback
# (see wrap_tool_call) — a message has to actually read as a follow-up to
# the current topic, not merely fail to classify on its own, before an
# earlier-established category is trusted to override a decline.
CONTINUATION_RE = re.compile(
    r"\b(alternative|another|else|different|instead|other|more)\b",
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


def _established_category(session_id: str) -> str | None:
    """The fever/cold category of whatever product list was last shown this
    session, if any — used to keep a follow-up turn anchored to the same
    category rather than trusting the model to remember or re-derive it."""
    last = store.get_last_products(session_id)
    if not last:
        return None
    return "fever" if last[0]["id"].startswith("fev") else "cold"


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
# ("wet"/"chesty") cough per its own knowledge-base entry; fever products
# split sharply between adult and child dosing (fev-004 is the only
# pediatric option) — the wrong pick there is a real dosing mismatch, not
# just a suboptimal one. Each "branch" maps a set of answer words to either
# a fixed reply (e.g. "see a pharmacist", no product fits) or a list of
# product_ids to recommend — one product renders as a direct "would you like
# to order this?", more than one renders as a normal multi-option list.
# Extensible: add more entries here for other symptoms worth narrowing down
# before recommending, following the same {qualifiers, question, branches} shape.
CLARIFYING_QUESTIONS = {
    "cough": {
        "qualifiers": ("dry", "wet", "chesty", "productive", "phlegm", "mucus", "sputum"),
        "question": (
            "Is your cough dry, or is it bringing up mucus (a chesty/productive "
            "cough)? Our cough product is a dry-cough suppressant only, so this "
            "makes sure it's actually the right fit."
        ),
        "branches": [
            {"answers": ("dry",), "product_ids": ("col-004",)},
            {
                "answers": ("wet", "chesty", "productive", "phlegm", "mucus", "sputum"),
                "product_ids": (),
                "reply": (
                    "Our only cough product (Cough Suppressant Syrup) is a dry-cough "
                    "suppressant, so it isn't the right fit for a productive/chesty "
                    "cough — I'd recommend checking with a pharmacist for that instead. "
                    "Is there anything else I can help with?"
                ),
            },
        ],
    },
    "fever": {
        "qualifiers": ("child", "kid", "kids", "baby", "infant", "toddler", "adult", "myself", "grown"),
        "question": (
            "Is this fever for a child, or for an adult/yourself? Our child and "
            "adult fever products use different dosing, so this makes sure the "
            "recommendation is the right fit."
        ),
        "branches": [
            {"answers": ("child", "kid", "kids", "baby", "infant", "toddler"), "product_ids": ("fev-004",)},
            {"answers": ("adult", "myself", "grown", "me"), "product_ids": ("fev-001", "fev-002", "fev-003")},
        ],
    },
    "cold": {
        # Unlike fever, there's no pediatric-formulated option at all in our
        # cold catalog (antihistamine, decongestant, and combo relief are all
        # adult products) — a child mention isn't a narrowing question, it's
        # a real "nothing here fits" case, same shape as cough's wet-cough
        # branch.
        "qualifiers": ("child", "kid", "kids", "baby", "infant", "toddler", "adult", "myself", "grown"),
        "question": (
            "Is this cold for a child, or for an adult/yourself? None of our "
            "cold products are formulated or dosed for children, so this "
            "makes sure we don't recommend something unsuitable."
        ),
        "branches": [
            {
                "answers": ("child", "kid", "kids", "baby", "infant", "toddler"),
                "product_ids": (),
                "reply": (
                    "None of our cold products (antihistamine, decongestant, or "
                    "combination relief) are formulated or dosed for children — "
                    "I'd recommend checking with a pharmacist or pediatrician "
                    "instead. Is there anything else I can help with?"
                ),
            },
            {
                "answers": ("adult", "myself", "grown", "me"),
                "product_ids": ("col-001", "col-002", "col-003", "col-004"),
            },
        ],
    },
}


# "fever" and "cold" are checked against tools.classify_categories rather
# than a literal substring of the trigger word — a message like "high
# temperature" (no literal "fever") or "runny nose" (no literal "cold")
# already classifies into these categories via FEVER_KEYWORDS/COLD_KEYWORDS,
# and skipping the clarifying question for a wording that doesn't happen to
# contain the exact trigger word defeats the whole point of asking it.
# Observed failure: "i think i have running nose and high temperature" went
# straight to an unfiltered product list — including the pediatric
# Children's Paracetamol Syrup alongside adult-only Ibuprofen — because the
# literal string "fever" never appeared, even though the message plainly
# classifies as fever. "cough" stays a literal check since it's a specific
# cold sub-symptom, not itself a classify_categories() category.
_CATEGORY_TRIGGERS = {"fever", "cold"}


def _trigger_matches(trigger: str, source_text_lower: str, categories: set[str]) -> bool:
    if trigger in _CATEGORY_TRIGGERS:
        return trigger in categories
    return trigger in source_text_lower


def _needs_clarification(source_text: str) -> tuple[str, str] | None:
    """Returns (trigger, question) if source_text mentions an ambiguous
    symptom without its qualifier already present, else None."""
    s = source_text.lower()
    categories = set(tools.classify_categories(source_text))
    for trigger, rule in CLARIFYING_QUESTIONS.items():
        if _trigger_matches(trigger, s, categories) and not any(q in s for q in rule["qualifiers"]):
            return trigger, rule["question"]
    return None


def _render_products(products: list[dict], session_id: str) -> str:
    store.set_last_products(session_id, products)
    if len(products) == 1:
        product = products[0]
        store.set_last_recommended_product(session_id, product["id"])
        return (
            f"{product['name']} would be a good fit — {product['description']}\n\n"
            f"Would you like to order this?\n\n{guardrails.DISCLAIMER}"
        )
    lines = ["Here's what I'd recommend:", ""]
    for p in products:
        lines.append(f"- {p['name']} ({p['id']}) — {p['description']}")
    lines.append("")
    lines.append("Which one would you like to try?")
    lines.append("")
    lines.append(guardrails.DISCLAIMER)
    return "\n".join(lines)


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
    for branch in rule["branches"]:
        if any(a in s for a in branch["answers"]):
            if not branch["product_ids"]:
                return branch["reply"]
            products = [p for p in (store.find_product(pid) for pid in branch["product_ids"]) if p]
            return _render_products(products, session_id)
    return None


def _resolve_prequalified_clarification(source_text: str, session_id: str) -> str | None:
    """Handles a qualifier arriving in the same message as the trigger word
    (see run_turn) — same branch-matching as resolve_clarification, just
    entered directly from the original message instead of a follow-up
    answer to a question that was actually asked."""
    s = source_text.lower()
    categories = set(tools.classify_categories(source_text))
    for trigger, rule in CLARIFYING_QUESTIONS.items():
        if _trigger_matches(trigger, s, categories) and any(q in s for q in rule["qualifiers"]):
            return resolve_clarification(trigger, source_text, session_id)
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
            f"catalog products (don't call lookup_symptom again): {result}. Present "
            f"these as a recommendation and ask if they'd like to order one — do NOT "
            f"call start_order yet unless this message already contains clear ordering "
            f"language (e.g. 'order X', 'buy X'); wait for their next message to confirm.]"
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

TOOL_ROUND_LIMIT_REPLY = "Sorry, I'm having trouble completing that request right now — could you rephrase?"


def _build_tools(session_id: str) -> list:
    """Builds a fresh set of LangChain tools for this one turn, closing over
    session_id — start_order needs it to actually place/defer an order, and
    it's the caller's context (never something the model itself supplies),
    same as in the raw-ollama tool schema this replaces."""

    @tool
    def lookup_symptom(symptom: str) -> str:
        """Look up OTC medicine recommendations from the fixed catalog for a fever
        or cold symptom. ONLY covers fever and cold (e.g. fever, headache-with-fever,
        runny nose, cough, congestion, sore throat, chills). Returns an empty result
        for anything else — do not call this for symptoms unrelated to fever/cold;
        instead tell the user this demo can't help with that."""
        return tools.lookup_symptom(symptom)

    @tool
    def get_saved_address(user_id: str) -> str:
        """Retrieve the saved shipping address for the current user, if one exists.
        Always call this before asking the user for an address and before placing an order."""
        return tools.get_saved_address(user_id)

    @tool
    def save_address(user_id: str, address: str) -> str:
        """Save a shipping address for the current user so future orders don't need
        to ask again. Call this only after the user has explicitly provided their
        address in chat."""
        return tools.save_address(user_id, address)

    @tool
    def start_order(product_id: str, quantity: int = 1) -> str:
        """Start placing an order for a catalog product once the user has
        confirmed they want it. The product_id MUST be one returned by a prior
        lookup_symptom call — never invent a product_id or order something
        outside the fever/cold catalog. This never completes the order
        immediately, even if an address is already on file: it always asks the
        user to confirm their address first (people move — never assume a
        saved address is still correct), or asks for one if there isn't any on
        file yet. Their next message resolves whichever question this asked.
        You never need to call this (or any other tool) again for the same order."""
        return tools.start_order(product_id, session_id, quantity)

    @tool
    def lookup_medicine_info(query: str) -> str:
        """Look up dosage, common side effects, or warnings for a catalog medicine
        from our knowledge base (retrieval-augmented — this returns real excerpts
        to quote/cite, not a free-form answer). Only covers the 8 fever/cold
        products in our catalog. Returns an empty result for anything else."""
        return tools.lookup_medicine_info(query)

    @tool
    def decline_out_of_scope() -> str:
        """Call this when the user's request is not about fever/cold symptoms,
        medicine suggestions, or placing a fever/cold OTC order — e.g. general
        chit-chat, coding help, unrelated medical conditions or injuries, or
        anything else outside this demo's narrow scope. Do not attempt to answer
        the off-topic request yourself; just call this tool."""
        # Never actually reached — _GuardrailMiddleware.wrap_tool_call always
        # short-circuits this tool itself (either redirecting to a real
        # lookup_symptom call, or forcing the canned decline reply), the same
        # way the old manual loop special-cased it rather than dispatching it
        # through a real implementation.
        return "declined — told the user this is out of scope"

    return [lookup_symptom, get_saved_address, save_address, start_order, lookup_medicine_info, decline_out_of_scope]


class _GuardrailMiddleware(AgentMiddleware):
    """Re-implements every tool-call-time and final-reply-time guardrail that
    used to live inline in the manual tool-calling loop, as two hooks:
    `wrap_tool_call` (pre-execution: block, redirect, or correct a call
    before it runs) and `after_agent` (post-execution: override the model's
    final text). Constructed fresh per turn — see run_turn — since every
    guardrail here depends on this specific turn's user_text/session_id/
    grounding state, not anything reusable across turns."""

    def __init__(
        self,
        session_id: str,
        user_text: str,
        image_b64: str | None,
        had_shown_recommendation: bool,
        symptom_lookup_grounded: bool,
        turn_state: dict,
    ):
        super().__init__()
        self.session_id = session_id
        self.user_text = user_text
        self.image_b64 = image_b64
        self.had_shown_recommendation = had_shown_recommendation
        self.symptom_lookup_grounded = symptom_lookup_grounded
        self.turn_state = turn_state

    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        args = request.tool_call["args"]
        call_id = request.tool_call["id"]
        session_id = self.session_id
        user_text = self.user_text

        try:
            if name == "decline_out_of_scope":
                # Mirrors the ungrounded-lookup_symptom guard, in reverse: the
                # model declined a message that's actually grounded (either
                # the text itself classifies, or it's a continuation of a
                # category already established this session) — e.g. "any
                # alternative?" got declined outright despite fever context
                # from the same session. Redirect to a real lookup instead of
                # trusting the decline, rather than ending the turn wrong.
                #
                # The established-category fallback is deliberately gated on
                # CONTINUATION_RE, not trusted for any unclassified text —
                # "who is narendra modi" doesn't classify either, but it also
                # isn't remotely a continuation of an earlier cold/fever
                # topic, and a bare "a category exists somewhere in this
                # session" was wrongly overriding a CORRECT decline for it,
                # forcing an unrelated product list into the reply instead.
                category = (tools.classify(user_text) if user_text else None) or (
                    _established_category(session_id)
                    if user_text and CONTINUATION_RE.search(user_text)
                    else None
                )
                if category:
                    print(f"[GUARD] blocked incorrect decline_out_of_scope on grounded input: {user_text!r}")
                    result = tools.lookup_symptom(category)
                    _remember_products(session_id, result)
                    return ToolMessage(content=result, tool_call_id=call_id, name=name)
                # Not grounded — let the model take one more (discarded) turn
                # reacting to this tool result; after_agent forces the canned
                # OUT_OF_SCOPE_REPLY regardless of what it says next, the
                # same end result the old loop got by returning immediately.
                self.turn_state["declined_forced"] = True
                return ToolMessage(content="declined — told the user this is out of scope", tool_call_id=call_id, name=name)

            if name == "lookup_symptom":
                if user_text and INFO_QUESTION_RE.search(user_text):
                    print(f"[GUARD] redirected misrouted lookup_symptom to lookup_medicine_info: {user_text!r}")
                    result = tools.lookup_medicine_info(user_text)
                    return ToolMessage(content=result, tool_call_id=call_id, name=name)

                text_classifies = bool(user_text) and tools.classify(user_text) is not None
                established = _established_category(session_id)
                if not (bool(self.image_b64) or text_classifies) and established:
                    # The current message's own text doesn't establish any
                    # category — the only reason this call isn't blocked as
                    # ungrounded below is an established category from
                    # earlier this session. That justifies reusing THAT
                    # category, never switching to a different one the model
                    # invented. Observed failure: asked "last one" right
                    # after a fever product list, the model called
                    # lookup_symptom(symptom="cold") with nothing in the
                    # message suggesting cold at all.
                    requested = tools.classify(args.get("symptom", ""))
                    if requested != established:
                        print(
                            f"[GUARD] blocked lookup_symptom category switch "
                            f"(requested={args.get('symptom')!r}, established={established!r}): {args}"
                        )
                    result = tools.lookup_symptom(established)
                    _remember_products(session_id, result)
                    return ToolMessage(content=result, tool_call_id=call_id, name=name)

                if not self.symptom_lookup_grounded:
                    print(f"[GUARD] blocked ungrounded lookup_symptom call: {args}")
                    self.turn_state["blocked_ungrounded_lookup"] = True
                    result = json.dumps({
                        "matched": False,
                        "message": (
                            "This message doesn't describe a fever or cold symptom. Do "
                            "not present catalog products for it — call "
                            "decline_out_of_scope instead."
                        ),
                    })
                    return ToolMessage(content=result, tool_call_id=call_id, name=name)

                response = handler(request)
                print(f"[TOOL CALL] {name}({args}) -> {response.content}")
                _remember_products(session_id, response.content)
                return response

            if name == "start_order":
                order_intent_this_message = bool(user_text) and ORDER_INTENT_RE.search(user_text)
                if not self.had_shown_recommendation and not order_intent_this_message:
                    print(f"[GUARD] blocked premature start_order before user confirmed intent: {args}")
                    result = json.dumps({
                        "order_placed": False,
                        "message": (
                            "The user has not confirmed they want to order this yet "
                            "— do not call start_order. Instead, present the "
                            "recommended product in plain text and ask \"Would you "
                            "like to order this?\", then stop without calling any "
                            "tool this turn."
                        ),
                    })
                    return ToolMessage(content=result, tool_call_id=call_id, name=name)

                response = handler(request)
                print(f"[TOOL CALL] {name}({args}) -> {response.content}")
                try:
                    parsed_order = json.loads(response.content)
                except (TypeError, ValueError):
                    parsed_order = {}
                if parsed_order.get("order_placed"):
                    self.turn_state["real_order_placed"] = True
                    self.turn_state["completed_order"] = parsed_order
                    store.clear_last_recommended_product(session_id)
                elif "error" not in parsed_order:
                    self.turn_state["deferred_order"] = parsed_order
                if parsed_order.get("email_sent"):
                    self.turn_state["real_email_sent"] = True
                return response

            if name == "save_address":
                response = handler(request)
                print(f"[TOOL CALL] {name}({args}) -> {response.content}")
                try:
                    if json.loads(response.content).get("saved"):
                        self.turn_state["real_address_saved"] = True
                except (TypeError, ValueError):
                    pass
                return response

            response = handler(request)
            print(f"[TOOL CALL] {name}({args}) -> {response.content}")
            return response
        except Exception as exc:
            print(f"[TOOL ERROR] {name}({args}): {exc}")
            return ToolMessage(content=f"Error calling {name}: {exc}", tool_call_id=call_id, name=name)

    def after_agent(self, state, runtime) -> dict | None:
        last = state["messages"][-1]
        reply_text = last.content or ""
        session_id = self.session_id

        if self.turn_state.get("blocked_ungrounded_lookup") or self.turn_state.get("declined_forced"):
            # We already know structurally that this turn's lookup was
            # invalid, or that a decline should stand — don't leave the
            # final wording up to the model regardless of what it said in
            # reaction to the tool result.
            return {"messages": [AIMessage(content=guardrails.OUT_OF_SCOPE_REPLY, id=last.id)]}

        if self.turn_state.get("completed_order"):
            # A real order was placed this turn — render the confirmation
            # from the actual order data instead of the model's own
            # phrasing. Observed failure: the model sometimes free-generates
            # a vague confirmation that omits the order id, price, and
            # address the user needs, even though the order itself was
            # genuinely placed.
            final = guardrails.build_order_confirmation(self.turn_state["completed_order"])
            return {"messages": [AIMessage(content=final, id=last.id)]}

        leaked_tool = guardrails.leaked_tool_intent(reply_text, TOOL_NAMES)
        if leaked_tool == "decline_out_of_scope":
            print(f"[GUARD] converted leaked decline intent to real decline: {reply_text!r}")
            final = guardrails.OUT_OF_SCOPE_REPLY
        elif leaked_tool == "lookup_symptom":
            recovered = guardrails.recover_leaked_lookup(reply_text)
            if recovered:
                print(f"[GUARD] recovered leaked lookup_symptom call: {reply_text!r}")
                _remember_products(session_id, recovered[1])
                final = recovered[0]
            else:
                print(f"[GUARD] blocked leaked tool intent (lookup_symptom, unrecoverable): {reply_text!r}")
                final = guardrails.FAKE_COMPLETION_GUARD_REPLY
        elif leaked_tool:
            print(f"[GUARD] blocked leaked tool intent ({leaked_tool}): {reply_text!r}")
            final = guardrails.FAKE_COMPLETION_GUARD_REPLY
        elif self.turn_state.get("deferred_order"):
            # Never trust the model's own phrasing here — see
            # guardrails.reply_for_deferred_order for the two distinct
            # observed failures this avoids.
            final = guardrails.reply_for_deferred_order(self.turn_state["deferred_order"])
        else:
            guardrails.remember_recommended_product(session_id, reply_text)
            guard_reply = guardrails.check_unverified_completion(
                reply_text,
                self.turn_state.get("real_order_placed", False),
                self.turn_state.get("real_email_sent", False),
                self.turn_state.get("real_address_saved", False),
            )
            final = guard_reply or reply_text

        return {"messages": [AIMessage(content=final, id=last.id)]}


def _invoke_agent_with_retry(agent_graph, payload: dict, run_config: dict):
    """One retry before giving up — Ollama running locally on CPU occasionally
    has a slow/failed first call under load; a single retry smooths that over
    without masking a genuinely dead service. A recursion-limit hit is a real
    cap, not a transient failure, so it's never retried."""
    last_exc = None
    for attempt in range(2):
        try:
            return agent_graph.invoke(payload, config=run_config)
        except GraphRecursionError:
            raise
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

        # The qualifier can arrive pre-answered in the same message ("my baby
        # has a fever") — _needs_clarification correctly skips asking again,
        # but without this, the message then fell through to a plain
        # lookup_symptom call that returns every product in the category
        # unfiltered, including ones the qualifier just ruled out (e.g. a
        # bare "2" after "my baby has a fever" could select the adult-only
        # Extra Strength product). Resolving directly against the same
        # branch logic keeps this deterministic and consistent with the
        # question-then-answer path instead of leaving it to the model.
        prequalified_reply = _resolve_prequalified_clarification(user_text, session_id)
        if prequalified_reply:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": prequalified_reply})
            return prequalified_reply, messages

    # Snapshot BEFORE _build_user_text runs — it records this turn's own
    # detected products via _remember_products, which would otherwise make
    # "a recommendation exists" look true even on the very first mention of a
    # symptom, defeating the start_order guard below.
    had_shown_recommendation = bool(store.get_last_products(session_id)) or bool(
        store.get_last_recommended_product(session_id)
    )

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

    # Tracks whether a *real* start_order success (and, separately, a real
    # email send) happened anywhere in this turn — see
    # guardrails.check_unverified_completion for why this can't just be
    # inferred from "no tool_calls this iteration". Mutated by
    # _GuardrailMiddleware, which closes over this same dict.
    turn_state: dict = {
        "real_order_placed": False,
        "real_email_sent": False,
        "real_address_saved": False,
        "completed_order": None,
        "deferred_order": None,
        "blocked_ungrounded_lookup": False,
        "declined_forced": False,
    }

    agent_tools = _build_tools(session_id)
    middleware = [
        _GuardrailMiddleware(
            session_id=session_id,
            user_text=user_text,
            image_b64=image_b64,
            had_shown_recommendation=had_shown_recommendation,
            symptom_lookup_grounded=symptom_lookup_grounded,
            turn_state=turn_state,
        )
    ]
    # Rebuilt fresh every turn — tools and middleware both close over this
    # turn's session_id/user_text/grounding state, which can't be cached
    # across turns.
    agent_graph = create_agent(_chat_model, tools=agent_tools, system_prompt=SYSTEM_PROMPT, middleware=middleware)

    run_config = {"recursion_limit": MAX_TOOL_ROUNDS * 2 + 4}

    try:
        result = _invoke_agent_with_retry(agent_graph, {"messages": messages}, run_config)
    except GraphRecursionError:
        messages.append({"role": "assistant", "content": TOOL_ROUND_LIMIT_REPLY})
        return TOOL_ROUND_LIMIT_REPLY, messages
    except Exception:
        messages.append({"role": "assistant", "content": OLLAMA_UNAVAILABLE_REPLY})
        return OLLAMA_UNAVAILABLE_REPLY, messages

    reply_text = result["messages"][-1].content or ""
    messages.append({"role": "assistant", "content": reply_text})
    return reply_text, messages
