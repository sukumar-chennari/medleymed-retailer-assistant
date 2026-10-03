from pydantic import BaseModel, Field

# Real bug: none of these fields had any bound, so a direct API call could
# store a multi-megabyte message or a megabyte-long session_id (used as a DB
# row key) with a 200, growing session_state/metrics_events without limit
# and sending the whole thing to the LLM. The real UI never gets near any of
# these (UUID session ids, short typed text, one photo), so the limits only
# affect abusive/buggy callers. image_b64 is generous on purpose — the
# frontend sends the raw photo with no resizing, and a large phone photo is
# ~10MB of base64.
MAX_SESSION_ID_LEN = 64
MAX_TEXT_LEN = 4000
MAX_IMAGE_B64_LEN = 15_000_000
# Empty is allowed: the frontend sends file.type, which can be "" for an
# unrecognized file, and agent.py already falls back to image/jpeg then.
IMAGE_MEDIA_TYPE_PATTERN = r"^(image/(jpeg|png|webp|gif|heic|heif))?$"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=MAX_SESSION_ID_LEN)
    text: str = Field(default="", max_length=MAX_TEXT_LEN)
    image_b64: str | None = Field(default=None, max_length=MAX_IMAGE_B64_LEN)
    image_media_type: str | None = Field(default=None, pattern=IMAGE_MEDIA_TYPE_PATTERN)


class ChatResponse(BaseModel):
    reply: str


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=MAX_SESSION_ID_LEN)
    rating: str
    reply_snippet: str = Field(default="", max_length=500)
