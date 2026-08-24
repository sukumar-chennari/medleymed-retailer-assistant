"""Safety-net logic that runs on top of the agent's own decisions.

Everything here exists because the small local model (llama3.2, 3B) proved
unreliable at things a much larger hosted model would get right on its own:
narrating a tool call as text instead of invoking it, claiming an order/email
completed with no tool call behind it, or declining a plain greeting as
out-of-scope. None of this is prompt-only — each guard is a deterministic
check applied to the model's actual output or actual tool results.
"""

import json
import re

from app import store, tools

OUT_OF_SCOPE_REPLY = "This demo only handles fever and cold OTC guidance — I can't help with that here."

FAKE_COMPLETION_GUARD_REPLY = (
    "Let's make sure that actually goes through — what symptom is this for "
    "(fever or cold)? That way I can look up the right product and place a "
    "real order for you."
)

DISCLAIMER = (
    "This is general OTC guidance, not a medical diagnosis — consult a doctor if "
    "symptoms persist or worsen."
)


def claims_order_placed(reply_text: str) -> bool:
    t = reply_text.lower()
    return "order" in t and any(k in t for k in ("placed", "confirmed", "shipped", "is on its way"))


def claims_email_sent(reply_text: str) -> bool:
    t = reply_text.lower()
    return any(
        k in t
        for k in (
            "confirmation email", "email has been sent", "email has also been sent",
            "will send you an email", "sent you an email", "email sent",
        )
    )


def check_unverified_completion(
    reply_text: str, real_order_placed_this_turn: bool, real_email_sent_this_turn: bool
) -> str | None:
    """Catches the model narrating an order/email as done ("I've saved your
    address... order has been placed") without a real tool call backing it up
    this turn. Returns a replacement reply if the claim is unverified, else
    None. The two claims are tracked separately: a real order_placed:true
    doesn't make an accompanying "email has been sent" claim true too — the
    start_order result can report order_placed:true and email_sent:false in
    the same result (no email on file), and the model has been observed
    claiming the email was sent anyway."""
    unverified_order = claims_order_placed(reply_text) and not real_order_placed_this_turn
    unverified_email = claims_email_sent(reply_text) and not real_email_sent_this_turn
    if unverified_order or unverified_email:
        print(
            f"[GUARD] blocked unverified completion claim "
            f"(order={unverified_order}, email={unverified_email}): {reply_text!r}"
        )
        return FAKE_COMPLETION_GUARD_REPLY
    return None


def leaked_tool_intent(reply_text: str, tool_names: set[str]) -> str | None:
    """Catches the model narrating a tool call as text instead of using the
    real mechanism — e.g. "I'll call decline_out_of_scope to end this
    conversation" or raw {"name": "start_order", ...} / {"type": "function",
    "function": {"name": "lookup_symptom", ...}} JSON in either compact or
    spaced form. Returns the tool name it was trying to invoke, if
    recognizable, so the caller can perform the real action instead of
    leaking the planning text to the user."""
    t = reply_text.lower()
    narration = any(p in t for p in ("i'll call", "i will call", "calling the", "call the tool"))
    for name in tool_names:
        if name not in t:
            continue
        if narration or re.search(r'["\']name["\']\s*:\s*["\']' + re.escape(name) + r'["\']', t):
            return name
    return None


def recover_leaked_lookup(reply_text: str) -> tuple[str, str] | None:
    """The leaked JSON for lookup_symptom conveniently still contains the
    real symptom the model meant to look up — so rather than just suppressing
    the leak (a dead-end "please rephrase" that loses the user's answer),
    perform the real lookup ourselves and return a properly phrased reply.
    Returns (reply_text, raw_lookup_result_json) or None if unrecoverable."""
    match = re.search(r'["\']symptom["\']\s*:\s*["\']([^"\']+)["\']', reply_text, re.IGNORECASE)
    if not match:
        return None
    result_json = tools.lookup_symptom(match.group(1))
    result = json.loads(result_json)
    if not result.get("matched"):
        return OUT_OF_SCOPE_REPLY, result_json
    lines = ["Here's what I'd recommend:", ""]
    for p in result["products"]:
        lines.append(f"- {p['name']} ({p['id']}) — {p['description']}")
    lines.append("")
    lines.append("Would you like to order one of these?")
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines), result_json


PRODUCT_ID_RE = re.compile(r"\b(?:fev|col)-\d{3}\b", re.IGNORECASE)


def remember_recommended_product(session_id: str, reply_text: str) -> None:
    """Tracks the single product the assistant just recommended (parsed out
    of its own reply) so a later bare "yes"/"ok" confirmation — the natural
    way people respond to "would you like to order this?" — can be resolved
    deterministically instead of trusting the model to remember and act on
    it reliably across another turn."""
    matches = {m.lower() for m in PRODUCT_ID_RE.findall(reply_text)}
    if len(matches) == 1:
        product_id = next(iter(matches))
        if store.find_product(product_id):
            store.set_last_recommended_product(session_id, product_id)


GREETING_REPLY = "Hi! I can help with fever or cold symptoms, or a photo of a medicine label — what's going on?"

BYE_REPLY = "Take care! Come back anytime you have fever or cold questions."

# Word-level (not exact-phrase) matching — a fixed phrase list is too brittle
# for casual variants like "hey whatup" or "yo whats good". Any message that
# (a) contains no recognizable fever/cold content per tools.classify, (b) is
# short, and (c) contains one of these words is treated as a pleasantry. (a)
# is what stops this from swallowing a real request like "hi, I have a fever".
GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "helo", "hey", "hiya", "yo", "sup",
    "whatup", "whatsup", "wassup", "morning", "evening", "buddy",
}
BYE_WORDS = {"thanks", "thank", "thx", "ty", "bye", "goodbye", "cya", "cheers"}

# Idiom-level phrases ("how are you") use generic words (how/are/you) that
# would cause false positives if added to GREETING_WORDS individually, so
# they're matched as whole phrases instead — checked as a substring of the
# normalized text so "how are you bro?" still matches.
GREETING_PHRASES = {
    "how are you", "how r u", "how are u", "hows it going", "how's it going",
    "how you doing", "how you doin", "how ya doing", "whats good",
    "what's good", "how is it going", "how's everything",
}
MAX_PLEASANTRY_WORDS = 6


def deterministic_pleasantry_reply(text: str) -> str | None:
    """Greeting/pleasantry handling relies on a rule the model followed
    inconsistently in testing (a plain "hi" sometimes still triggered
    decline_out_of_scope, a 3B-model reliability gap, not a prompt-wording
    problem). Short-circuiting known pleasantries in code guarantees
    consistent behavior instead of hoping the model applies the instruction."""
    if tools.classify(text) is not None:
        return None  # real symptom/medicine content — let the normal flow handle it

    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return None

    normalized = " ".join(words)
    if any(phrase in normalized for phrase in GREETING_PHRASES):
        return GREETING_REPLY

    if len(words) > MAX_PLEASANTRY_WORDS:
        return None
    if any(w in BYE_WORDS for w in words):
        return BYE_REPLY
    if any(w in GREETING_WORDS for w in words):
        return GREETING_REPLY
    return None
