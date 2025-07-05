import os
import logging
from fastapi import APIRouter, HTTPException, Request, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
from sqlalchemy import func
from datetime import datetime
import time
import psutil

from ..schemas.models import ChatInteraction, ChatRequest, ChatResponse, ClearResponse
from ..database import db
from ..const.constants import VERSION

# Configure logging for this router
logger = logging.getLogger(__name__)

# Router initialization
router = APIRouter(tags=["Admin section"])

# Templates initialization - make sure path is relative to main app
templates = Jinja2Templates(directory="app/templates")

# Admin security initialization
api_key_header = APIKeyHeader(name="X-API-Key")


# Initialize router with required dependencies
def init_router(config_instance, chatbot_instance, start_time):
    """Initialize the routers with required dependencies"""
    global config, chatbot, START_TIME
    config = config_instance
    chatbot = chatbot_instance
    START_TIME = start_time


# API key verification for protected endpoints
async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key for protected endpoints"""
    if api_key != config.admin_api_key:
        raise HTTPException(
            status_code=403, detail="Could not validate API key (wrong API key)"
        )
    return api_key


# API endpoints
@router.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot and get a response
    """
    try:
        # Use provided session_id or get current one from chatbot
        session_id = request.session_id or chatbot.current_session_id
        response = chatbot.get_response(request.message, session_id)
        return ChatResponse(
            response=response, conversation_history=chatbot.get_conversation_history()
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/clear", response_model=ClearResponse)
async def clear_conversation():
    """
    Clear the conversation history
    """
    try:
        chatbot.clear_conversation()
        return ClearResponse(
            message="Conversation history cleared successfully", status=True
        )
    except Exception as e:
        logger.error(f"Error in clear endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to clear conversation history"
        )


@router.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get statistics about chat interactions including full conversation history"""
    try:
        with next(db.get_session()) as session:
            # Basic statistics
            total_interactions = session.query(ChatInteraction).count()
            avg_response_time = session.query(
                func.avg(ChatInteraction.response_time)
            ).scalar()
            error_rate = session.query(
                func.avg(ChatInteraction.error_occurred)
            ).scalar()

            # Category distribution
            category_counts = (
                session.query(ChatInteraction.category, func.count(ChatInteraction.id))
                .group_by(ChatInteraction.category)
                .all()
            )

            # Get all conversations with full details
            conversations = (
                session.query(ChatInteraction)
                .order_by(ChatInteraction.timestamp.desc())
                .all()
            )

            # Format conversations for response
            conversation_history = [
                {
                    "id": conv.id,
                    "session_id": conv.session_id,
                    "timestamp": conv.timestamp.isoformat(),
                    "category": conv.category,
                    "user_message": conv.user_message,
                    "bot_response": conv.bot_response,
                    "response_time": round(conv.response_time, 2),
                    "tokens_used": conv.tokens_used,
                    "error_occurred": bool(conv.error_occurred),
                }
                for conv in conversations
            ]

            return {
                "total_interactions": total_interactions,
                "average_response_time": (
                    round(avg_response_time, 2) if avg_response_time else 0
                ),
                "error_rate": round(error_rate * 100, 2) if error_rate else 0,
                "category_distribution": dict(category_counts),
                "conversations": conversation_history,
            }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint that monitors critical system components and chatbot status
    """
    try:
        # Check if OpenAI integration is working
        openai_status = chatbot.config.openai_api_key is not None

        # Check if vector store is accessible
        vector_store_status = chatbot.vector_store is not None

        # Get application metrics
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        uptime = time.time() - START_TIME
        conversation_count = len(chatbot.conversation_history)

        # Check templates directory
        templates_status = os.path.exists("app/templates") and os.path.exists(
            "app/templates/index.html"
        )

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": VERSION,
            "components": {
                "openai": "operational" if openai_status else "failed",
                "vector_store": "operational" if vector_store_status else "failed",
                "templates": "available" if templates_status else "missing",
            },
            "metrics": {
                "memory_usage_mb": round(memory_usage, 2),
                "uptime_seconds": round(uptime, 2),
                "active_conversations": conversation_count,
                "model_name": chatbot.config.model_name,
                "max_history": chatbot.config.max_history,
            },
            "config": {
                "temperature": chatbot.config.temperature,
                "chunk_size": chatbot.config.chunk_size,
                "chunk_overlap": chatbot.config.chunk_overlap,
                "top_k_results": chatbot.config.top_k_results,
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            },
        )
