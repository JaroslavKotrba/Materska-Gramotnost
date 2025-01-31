import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base

# Database initialization
port = int(os.getenv("PORT", 8000))  # Heroku
Base = declarative_base()  # Base


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
