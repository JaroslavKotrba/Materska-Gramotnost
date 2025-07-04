from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from typing import List
from pydantic import BaseModel
from ..database.database import Base


class ChatInteraction(Base):
    """Model for storing chat interactions"""

    __tablename__ = "chat_interactions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    response_time = Column(Float)
    category = Column(String(50))
    tokens_used = Column(Integer)
    error_occurred = Column(Integer, default=0)


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
