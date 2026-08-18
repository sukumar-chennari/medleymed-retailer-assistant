from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    text: str = ""
    image_b64: str | None = None
    image_media_type: str | None = None


class ChatResponse(BaseModel):
    reply: str
