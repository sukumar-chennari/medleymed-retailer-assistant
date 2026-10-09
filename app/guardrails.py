"""Safety-net logic that runs on top of the agent's own decisions.

Everything here exists because the small local model (llama3.2, 3B) proved
unreliable at things a much larger hosted model would get right on its own:
narrating a tool call as text instead of invoking it, claiming an order/email
completed with no tool call behind it, or declining a plain greeting as
out-of-scope. None of this is prompt-only — each guard is a deterministic
check applied to the model's actual output or actual tool results.
"""

import datetime
import json
import re

from app import store, tools

CANCELLATION_NOT_DONE_REPLY = (
    "I haven't cancelled anything yet. Tell me which order you'd like to cancel "
    "(for example, \"cancel my last order\") and I'll do it."
)
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


def build_order_confirmation(order: dict) -> str:
    """The single canonical order-confirmation template — every completion
    path (deterministic address/email follow-ups in main.py, and a real
    start_order success reached via the LLM tool-loop in agent.py) renders
    through this, rather than ever letting the model phrase this message
    itself. Observed failure mode this prevents: a real order succeeded, but
    the model's own free-text confirmation ("please allow 3-5 business days
    for delivery...") omitted the order id, price, and address — details the
    user actually needs — even though nothing was factually wrong."""
    email_note = (
        "A confirmation email has been sent."
        if order.get("email_sent")
        else "I don't have an email on file for you — reply with your email "
             "address if you'd like a confirmation sent."
    )
    quantity = order.get("quantity", 1)
    total = order.get("total_price_usd", order.get("price_usd"))
    clamp_note = f"\n{order['quantity_clamped']}" if order.get("quantity_clamped") else ""
    return (
        f"Order confirmed!\n\n"
        f"Order ID: {order['order_id']}\n"
        f"Product: {order['product_name']}\n"
        f"Quantity: {quantity}\n"
        f"Total: ${total}\n"
        f"Shipping to: {order['address']}\n"
        f"{clamp_note}\n"
        f"{email_note}\n\n{DISCLAIMER}"
    )


def build_cancellation_confirmation(order: dict) -> str:
    """Same rationale as build_order_confirmation: a cancellation actually
    happened, so it's rendered from the real result rather than left to the
    model's own phrasing — which would also risk tripping
    check_unverified_completion, since "your order has been cancelled" reads
    as order-completion language to that guard."""
    return (
        f"Order {order['order_id']} ({order['product_name']}) has been cancelled.\n\n"
        f"You're welcome to place a new order anytime."
    )


def reply_for_deferred_order(order: dict) -> str:
    """The correct thing to say after a start_order call that deferred
    (never completes synchronously — see tools.start_order) rather than
    completed. Shared by main.py (bare-selection/affirmative-triggered
    orders) and agent.py's run_turn — the latter now uses this
    unconditionally whenever a turn deferred, not just as a guard fallback,
    because trusting the model's own wording here failed in two distinct
    ways: even after start_order clearly returned order_placed:false, it
    sometimes wrote "I've placed your order..." (caught by the
    unverified-completion guard, which used to fall back to a generic,
    wrong-for-this-situation reply instead of this one); separately — and
    without tripping that guard at all, since it never claimed completion —
    it silently omitted that a requested quantity had been capped by our
    per-order limit, so the user only found out from the final order
    confirmation instead of before agreeing to it. quantity_clamped is the
    actual reason text from tools._clamp_quantity (or None), not a bare
    bool — a hardcoded "capped at our per-order limit" note here used to
    show up even when the real reason was a non-positive quantity being
    raised to 1, which isn't a limit at all."""
    clamp_note = f" (Note: {order['quantity_clamped']}.)" if order.get("quantity_clamped") else ""
    if order.get("needs_address_confirmation"):
        return (
            f"We have this address on file: {order['address_on_file']}. Should I "
            f"ship to this address, or would you like to give a different one?{clamp_note}"
        )
    return f"Sure! What's your shipping address so I can send that out?{clamp_note}"


_SENTENCE_BREAK_RE = re.compile(r"[.!?\n;]")
_NEGATION_RE = re.compile(r"\b(?:no|not|never|cannot|unable)\b|n['\u2019]t\b")
_NEXT_WORD_NEGATION_RE = re.compile(r"^\s*(?:(?:not|never)\b|\w+n['\u2019]t\b)")
_QUESTION_OPENER_RE = re.compile(
    r"^\s*(?:do|does|did|can|could|would|will|should|is|are|may|what|which|how|when|where|why|who)\b"
)
# Kept short on purpose: a wider window would let an unrelated "no"/"not"
# earlier in a long sentence exempt a genuine claim ("...no worries" AFTER a
# real claim never counts, only negation BEFORE the keyword does).
_NEGATION_WINDOW_CHARS = 25


def _has_unnegated_claim(t: str, keywords: tuple[str, ...]) -> bool:
    """True when any keyword appears as a genuine claim. Real bug this fixes:
    plain `k in t` substring matching made the three claims_* checks fire on
    honest, truthful replies — "You haven't placed any orders yet", "I can't
    place the order until you give me an address", "I haven't saved your
    address yet", "Do you send a confirmation email?" — replacing each with
    the canned "what symptom is this for?" reply. That matters most for the
    empty-order-history report: check_order_status with no orders grounds
    nothing, so the model's correct "no orders yet" answer was thrown away.

    A keyword occurrence is NOT a claim when, within its own sentence, it is
    preceded (within a few characters) by a negation, is immediately
    followed by one ("your order is not yet placed"), or sits in a question
    ("Do you send a confirmation email?"). Anything else still counts, so
    mixed replies ("No problem! Your order has been placed.") and the
    originally-observed fabrications remain blocked."""
    for keyword in keywords:
        start = t.find(keyword)
        while start != -1:
            end = start + len(keyword)
            sentence_start = 0
            for match in _SENTENCE_BREAK_RE.finditer(t, 0, start):
                sentence_start = match.end()
            next_break = _SENTENCE_BREAK_RE.search(t, end)
            sentence_end = next_break.start() if next_break else len(t)
            before = t[max(sentence_start, start - _NEGATION_WINDOW_CHARS) : start]
            sentence = t[sentence_start:sentence_end]
            is_question = bool(_QUESTION_OPENER_RE.match(sentence)) and (
                next_break is not None and t[next_break.start()] == "?"
            )
            negated = bool(_NEGATION_RE.search(before)) or bool(_NEXT_WORD_NEGATION_RE.match(t[end:sentence_end]))
            if not negated and not is_question:
                return True
            start = t.find(keyword, end)
    return False


def claims_order_placed(reply_text: str) -> bool:
    """Broad on purpose — a phrasing like "I'll go ahead and place the
    order... here's your order summary... once it's processed" describes a
    completed order without ever using the literal words "placed" or
    "confirmed" the original (narrower) version of this check looked for,
    and slipped through undetected along with a fabricated shipping address."""
    # "in order to" is just English, not an order mention — it used to satisfy
    # the bare "order" check and, with any "placed"/"confirmed" later in the
    # reply, block an ordinary health tip.
    t = reply_text.lower().replace("in order to", " ")
    return "order" in t and _has_unnegated_claim(
        t,
        (
            "placed", "confirmed", "shipped", "is on its way", "order summary",
            "place the order", "processing your order", "once it's processed",
            "order has been", "your order is",
            # A receipt-shaped reply. Real bug, seen live in a
            # conversation_eval.py run: asked to "reorder that", the model
            # skipped reorder_last entirely and wrote "Order ID: ord-0002 /
            # Product: ... / Quantity: 1 / Total: $6.29 / Shipping to: ..." —
            # an invented receipt for an order that never existed — and none
            # of the phrases above appear in it, so it sailed through. The
            # app's own confirmations never reach this check (they are
            # returned from after_agent before it runs) and an honest order
            # listing is grounded by check_order_status, so an ungrounded
            # "Order ID:" label is a fabricated receipt.
            "order id:",
        ),
    )


def claims_order_cancelled(reply_text: str) -> bool:
    """A reply asserting an order WAS cancelled. Narrow on purpose — only
    past-tense/passive claim phrasings ("I've cancelled", "has been
    cancelled", "is now canceled"), so offers ("Would you like me to cancel
    it?"), refusals ("I can't cancel that"), negations ("hasn't been
    cancelled") and status listings ("status: cancelled") are not claims."""
    return _has_unnegated_claim(
        reply_text.lower(),
        (
            "i've cancelled", "i have cancelled", "i cancelled",
            "i've canceled", "i have canceled", "i canceled",
            "has been cancelled", "has been canceled",
            "successfully cancelled", "successfully canceled",
            "is now cancelled", "is now canceled",
            "order cancelled", "order canceled",
        ),
    )


def check_unverified_cancellation(reply_text: str, grounded_by_order_status: bool) -> str | None:
    """Real bug: cancel_order had a guard on the TOOL CALL (it only runs when
    this message asks to cancel) but nothing on the CLAIM, so with no real
    cancellation this turn "I've cancelled your order ord-0001." or "Done,
    order cancelled!" reached the customer unchanged — they believe the order
    is cancelled while it is still active and will ship. (A real
    cancellation never gets here: after_agent renders its own confirmation
    from the real result before this check runs.)

    grounded_by_order_status: check_order_status returned real orders this
    turn, so a reply listing them may legitimately say "cancelled". Returns a
    replacement reply, or None if the reply is fine."""
    if grounded_by_order_status or not claims_order_cancelled(reply_text):
        return None
    print(f"[GUARD] blocked unverified cancellation claim: {reply_text!r}")
    return CANCELLATION_NOT_DONE_REPLY


def claims_email_sent(reply_text: str) -> bool:
    return _has_unnegated_claim(
        reply_text.lower(),
        (
            "confirmation email", "email has been sent", "email has also been sent",
            "will send you an email", "sent you an email", "email sent",
        ),
    )


def claims_address_saved(reply_text: str) -> bool:
    """Added after a real observed failure: routing a message that doesn't
    look like an order/email/decline through a normal turn (see main.py's
    pending_product_id/pending_email_order_id fallbacks) reopened the exact
    hallucination this project already fixed once — the model claimed
    "I've saved your shipping address" for a bare place name with no digits
    (e.g. "hyderabad"), and nothing verified that claim, because only
    order-placed and email-sent claims were ever checked."""
    return _has_unnegated_claim(
        reply_text.lower(),
        (
            "saved your address", "saved your shipping address", "address has been saved",
            "address is now on file", "saved that address", "saved the address",
        ),
    )


def check_unverified_completion(
    reply_text: str,
    real_order_placed_this_turn: bool,
    real_email_sent_this_turn: bool,
    real_address_saved_this_turn: bool = False,
) -> str | None:
    """Catches the model narrating an order/email/address as done ("I've
    saved your address... order has been placed") without a real tool call
    backing it up this turn. Returns a replacement reply if the claim is
    unverified, else None. Each claim is tracked separately: a real
    order_placed:true doesn't make an accompanying "email has been sent"
    claim true too — the start_order result can report order_placed:true
    and email_sent:false in the same result (no email on file), and the
    model has been observed claiming the email was sent anyway."""
    unverified_order = claims_order_placed(reply_text) and not real_order_placed_this_turn
    unverified_email = claims_email_sent(reply_text) and not real_email_sent_this_turn
    unverified_address = claims_address_saved(reply_text) and not real_address_saved_this_turn
    if unverified_order or unverified_email or unverified_address:
        print(
            f"[GUARD] blocked unverified completion claim "
            f"(order={unverified_order}, email={unverified_email}, address={unverified_address}): {reply_text!r}"
        )
        return FAKE_COMPLETION_GUARD_REPLY
    return None


def leaked_tool_intent(reply_text: str, tool_names: set[str]) -> str | None:
    """Catches the model exposing internal tool-call mechanics as text
    instead of just using the real mechanism or answering in plain
    language — e.g. "I'll call decline_out_of_scope to end this
    conversation", raw {"name": "start_order", ...} JSON, or "you can start
    ordering this by calling start_order with the product_id 'fev-001'".

    Deliberately broad: any literal occurrence of a tool's snake_case name
    is treated as a leak, full stop, rather than requiring it to appear
    alongside a specific narration phrase ("I'll call", "calling the", ...).
    The narrower, phrase-gated version of this check missed "by calling
    start_order with the product_id" entirely, since that sentence never
    contains the literal substring "calling the" it was looking for — there
    is no legitimate reason a user-facing reply should ever contain a tool's
    internal name at all, regardless of the surrounding wording."""
    t = reply_text.lower()
    for name in tool_names:
        if re.search(r"\b" + re.escape(name) + r"\b", t):
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
    it reliably across another turn.

    Observed failure this guards against: a reply that names a product only
    by its human-readable name ("Paracetamol 500mg Tablets"), never its id
    ("fev-001"), left this mechanism blind — "yes" then fell through to the
    model's own memory, which ordered a *different* product (Extra Strength
    650mg) than the one actually shown. Falling back to matching by catalog
    product name closes that gap."""
    matches = {m.lower() for m in PRODUCT_ID_RE.findall(reply_text)}
    if not matches:
        text_lower = reply_text.lower()
        matches = {p["id"] for p in store.get_catalog() if p["name"].lower() in text_lower}
    if len(matches) == 1:
        product_id = next(iter(matches))
        if store.find_product(product_id):
            store.set_last_recommended_product(session_id, product_id)


GREETING_REPLY_TEMPLATE = "{opener} I can help with fever or cold symptoms, or a photo of a medicine label — what's going on?"
GREETING_REPLY = GREETING_REPLY_TEMPLATE.format(opener="Hi!")

BYE_REPLY = "Take care! Come back anytime you have fever or cold questions."

# Word-level (not exact-phrase) matching — a fixed phrase list is too brittle
# for casual variants like "hey whatup" or "yo whats good". Any message that
# (a) contains no recognizable fever/cold content per tools.classify, (b) is
# short, and (c) contains one of these words is treated as a pleasantry. (a)
# is what stops this from swallowing a real request like "hi, I have a fever".
GREETING_WORDS = {
    "hi", "hii", "hiii", "hio", "hello", "helo", "hey", "hiya", "yo", "sup",
    "whatup", "whatsup", "wassup", "morning", "morinig", "mornign",
    "afternoon", "evening", "buddy",
}
BYE_WORDS = {"thanks", "thank", "thx", "ty", "bye", "goodbye", "cya", "cheers"}

# Maps every recognized time-of-day word (including its typo variants) to the
# canonical period, so a message naming a period gets checked against the
# real current time rather than just echoed back — a user who types "good
# morning" during the afternoon gets corrected to "Good afternoon!" instead
# of the bot parroting the wrong time of day back at them.
_TIME_OF_DAY_WORDS = {
    "morning": "morning", "morinig": "morning", "mornign": "morning",
    "afternoon": "afternoon",
    "evening": "evening",
}


def _current_time_of_day_opener(now: datetime.datetime | None = None) -> str:
    """Buckets the current hour into a greeting opener. Late night (9pm-5am)
    has no natural "good ___" opener for an incoming chat, so it falls back
    to a plain "Hi!" rather than an odd "Good night!" said to someone
    arriving, not leaving."""
    hour = (now or datetime.datetime.now()).hour
    if 5 <= hour < 12:
        return "Good morning!"
    if 12 <= hour < 17:
        return "Good afternoon!"
    if 17 <= hour < 21:
        return "Good evening!"
    return "Hi!"


def _greeting_reply(words: list[str], now: datetime.datetime | None = None) -> str:
    named_period = next((w for w in words if w in _TIME_OF_DAY_WORDS), None)
    opener = _current_time_of_day_opener(now) if named_period else "Hi!"
    return GREETING_REPLY_TEMPLATE.format(opener=opener)


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
# The phrase check above ("how are you", ...) used to have NO length limit,
# unlike every word-based check below it, so any long message that merely
# contained one — "how are you going to help my brother with his knee pain
# after the match yesterday evening" — got the canned "Hi! I can help with
# fever or cold..." greeting and its real question was dropped. Phrases are
# allowed a little more room than bare words ("hey bro how are you doing
# today my friend"), but not an open-ended amount.
MAX_GREETING_PHRASE_WORDS = 8


# Real bug: a message that merely STARTS or ENDS with a pleasantry was answered
# with the canned greeting/goodbye and never reached the model — "thanks,
# order it", "bye, cancel my order", "ok thanks, change address", "hey whats
# the price" all got "Take care!" / "Hi! I can help with...", silently
# dropping the actual request. Any of these words means the message is asking
# for something, so it isn't a pure pleasantry. Deliberately a short list of
# unmistakable request words (not "yes"/"please"/"can", which are ordinary
# polite filler) so genuine pleasantries — "hi", "thanks a lot", "no thanks",
# "hows it going bro" — keep their deterministic reply.
_PLEASANTRY_REQUEST_WORDS = {
    "order", "orders", "ordering", "ordered", "reorder", "cancel", "cancelled", "canceled",
    "buy", "purchase", "price", "cost", "dose", "dosage", "address", "ship", "shipping",
    "email", "status", "refund", "change", "show", "tell", "want", "need",
    "what", "which", "why", "when", "where", "safe", "feeling",
}


def deterministic_pleasantry_reply(text: str, now: datetime.datetime | None = None) -> str | None:
    """Greeting/pleasantry handling relies on a rule the model followed
    inconsistently in testing (a plain "hi" sometimes still triggered
    decline_out_of_scope, a 3B-model reliability gap, not a prompt-wording
    problem). Short-circuiting known pleasantries in code guarantees
    consistent behavior instead of hoping the model applies the instruction.
    `now` is exposed only so tests can pin the clock; real calls omit it."""
    if tools.classify(text) is not None:
        return None  # real symptom/medicine content — let the normal flow handle it

    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return None
    if any(w in _PLEASANTRY_REQUEST_WORDS for w in words):
        return None  # a real request that happens to include a pleasantry

    normalized = " ".join(words)
    if len(words) <= MAX_GREETING_PHRASE_WORDS and any(phrase in normalized for phrase in GREETING_PHRASES):
        return GREETING_REPLY

    if len(words) > MAX_PLEASANTRY_WORDS:
        return None
    if any(w in BYE_WORDS for w in words):
        return BYE_REPLY
    if any(w in GREETING_WORDS for w in words):
        return _greeting_reply(words, now)
    return None
