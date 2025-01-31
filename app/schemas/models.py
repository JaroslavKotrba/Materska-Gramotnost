from typing import List
from pydantic import BaseModel


# Chat Related Models (pydantic models for request/response validation)
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: str = None


class ChatResponse(BaseModel):
    response: str
    conversation_history: List[Message]


class ClearResponse(BaseModel):
    message: str
    status: bool
