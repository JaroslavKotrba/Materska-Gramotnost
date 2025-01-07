# TO RUN APP

# conda env create -f environment.yml
# conda create -n mat_gram python=3.12
# conda env remove -n mat_gram

# uvicorn app.main:app --reload

# conda env export --name mat_gram > environment.yml
# pip list --format=freeze > requirements.txt
# psycopg2-binary==2.9.9 (Heroku Postgres)
# mysqlclient==2.2.1 (JAWSDB MySQL)

# conda env export --name mat_gram > environment.yml
# pip list --format=freeze > requirements.txt

import os
from typing import List, Dict
from datetime import datetime
import time
import psutil
import uuid
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Security, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Text,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Path
os.getcwd()

# FastAPI app initialization
app = FastAPI(
    title="Maternity Chatbot API",
    description="API for a Czech maternity advisory chatbot",
    version="1.0.0",
)

# CLIENT DOMAIN
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://jaroslavkotrba.com",
        "https://mat-gram-c3a70edf9532.herokuapp.com",  # Heroku
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start time for uptime tracking
START_TIME = time.time()

# Serve static files (CSS, JavaScript)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Serve Jinja2 for templates (index.html)
templates = Jinja2Templates(directory="app/templates")

# Admin security initialization
api_key_header = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key for protected endpoints"""
    if api_key != config.admin_api_key:  # Using the admin_api_key from config
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key


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


class Database:
    """Connect database"""

    def __init__(self):
        # Get JAWSDB_URL from Heroku
        database_url = os.getenv("JAWSDB_URL")

        if database_url:
            # We're on Heroku, use JawsDB MySQL
            self.engine = create_engine(database_url)
        else:
            # We're local, use SQLite
            self.engine = create_engine("sqlite:///chatbot.db")

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()


# Create database
db = Database()


# Pydantic models for request/response validation
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


class ChatbotConfig:
    """Class for chatbot configuration"""

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not available in .env")

        self.admin_api_key = os.getenv("ADMIN_API_KEY")
        if not self.admin_api_key:
            raise ValueError("ADMIN_API_KEY is not available in .env")

        self.model_name = "gpt-4o-mini"
        self.temperature = 0.4  # Controls randomness in responses
        self.chunk_size = 1000  # Size of text chunks for processing
        self.chunk_overlap = 200  # Overlap between chunks to maintain context
        self.top_k_results = 5  # Number of similar documents to retrieve
        self.max_history = 4  # Maximum number of conversation turns to remember


class CoreChatbot:
    """Core chatbot implementation.

    Integrates OpenAI's language models with FAISS vector storage
    for context-aware responses"""

    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        self.db = db
        self.current_session_id = str(uuid.uuid4())

        # Initialize embedding model for text vectorization
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=config.openai_api_key
        )

        # Set up the main language model for generating responses
        self.chat_model = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key,
        )

        # Initialize vector store and retrieval system
        self.vector_store = self._load_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.top_k_results}
        )

        # Set up conversation template and processing chains
        self.prompt = self._create_prompt_template()
        self.document_chain = create_stuff_documents_chain(
            llm=self.chat_model,
            prompt=self.prompt,
            document_variable_name="context",
        )
        self.retrieval_chain = create_retrieval_chain(
            self.retriever,
            self.document_chain,
        )

    def _load_vector_store(self) -> FAISS:
        """Load the FAISS vector store from disk.

        Contains pre-processed knowledge base.
        Raises Exception if loading fails."""
        try:
            return FAISS.load_local(
                "data/vector_store",
                self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"Error while loading of the vector store: {str(e)}")
            raise

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the conversation prompt template.

        Defines the chatbot's personality, expertise areas, and response guidelines
        for providing advice."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Jste přátelská česká chatbotka zaměřená na podporu a poradenství pro maminky s miminky.
                
                    Specializujete se na oblasti:
                    1. Péče o miminko:
                    - Kojení a výživa
                    - Spánek a denní režim
                    - Hygiena a péče o pokožku
                    - Psychomotorický vývoj dítěte
                    - Očkování a zdravotní péče
                    
                    2. Péče o maminku:
                    - Poporodní období a rekonvalescence
                    - Kojení a problémy s kojením
                    - Cvičení po porodu
                    - Psychická pohoda a prevence poporodní deprese
                    
                    3. Praktické rady:
                    - Výbavička pro miminko
                    - Kočárky a nosítka
                    - Bezpečnost a první pomoc
                    - Zavádění příkrmů
                    
                    4. Sociální podpora:
                    - Mateřská dovolená
                    - Rodičovský příspěvek
                    - Další dostupné dávky a podpora
                    
                    Používejte následující kontext k odpovědím na otázky. Odpovědi pište vykáním a spisovnou češtinou.
                    Odpovědi by měly být jasné, srozumitelné a vedené v přátelském, podporujícím tónu.
                    
                    Doporučené zdroje pro ověření informací:
                    - www.kojeni.cz pro informace o kojení
                    - www.pediatrie.cz pro zdravotní informace
                    - www.mpsv.cz pro informace o rodičovském příspěvku
                    
                    Pokud si nejste něčím jistí nebo je otázka mimo vaši oblast, přiznejte to a pokuste se i tak stručně odpovědět.
                    
                    Neopakujte zadanou otázku, vynechte zbytečné fráze jako že jste na hokejlogic.cz a dlouhé nabídky další pomoci.
                    """,
                ),
                (
                    "human",
                    """Předchozí konverzace: {chat_history}
                
                Kontext: {context}
                
                Aktuální otázka: {input}""",
                ),
            ]
        )

    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritical marks from Czech"""
        replacements = {
            "á": "a",
            "č": "c",
            "ď": "d",
            "é": "e",
            "ě": "e",
            "í": "i",
            "ň": "n",
            "ó": "o",
            "ř": "r",
            "š": "s",
            "ť": "t",
            "ú": "u",
            "ů": "u",
            "ý": "y",
            "ž": "z",
        }
        return "".join(replacements.get(c.lower(), c.lower()) for c in text)

    def _categorize_message(self, message: str) -> str:
        """Categorize the incoming message based on content"""
        categories = {
            "pece o miminko": ["miminko", "kojeni", "kojit", "spat", "vyvoj", "dite"],
            "pece o maminku": ["maminka", "porodni", "cviceni", "deprese"],
            "prakticke rady": ["vybavicka", "kocarek", "nositko", "prikrm"],
            "socialni podpora": ["materska", "rodicovska", "prispevek", "davky"],
        }

        message_normalized = self._remove_diacritics(message).lower()
        for category, keywords in categories.items():
            if any(keyword in message_normalized for keyword in keywords):
                return category
        return "ostatni"

    def format_chat_history(self) -> str:
        """Format the conversation history for context inclusion.

        Returns a string of the last N interactions based on max_history setting."""
        return "\n".join(
            [
                f"{'Uživatel' if msg['role'] == 'user' else 'Asistent'}: {msg['content']}"
                for msg in self.conversation_history[-self.config.max_history :]
            ]
        )

    def get_response(self, user_input: str, session_id: str = None) -> str:
        """Generate a response to user input using the language model.

        Updates conversation history and handles any errors during processing.
        Returns an error message if response generation fails."""

        # DB values
        session_id = session_id or self.current_session_id
        start_time = time.time()
        error_occurred = 0
        tokens_used = 0

        try:
            # Updating history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Getting answer
            response = self.retrieval_chain.invoke(
                {
                    "chat_history": self.format_chat_history(),
                    "input": user_input,
                }
            )

            # Remove asterisks from the response
            answer = response["answer"].replace("*", "")

            # Remove any "Dobrý den" variations from the answer
            answer = answer.replace("Dobrý den, ", "")
            answer = answer.replace("Dobrý den.", "")
            answer = answer.replace("Dobrý den!", "")
            answer = answer.replace("Dobrý den", "")  # Catch any remaining variants

            # Capitalize first letter of remaining text
            answer = answer.strip()  # Remove any leading/trailing whitespace
            if answer:  # Check if answer is not empty
                answer = (
                    answer[0].upper() + answer[1:]
                    if len(answer) > 1
                    else answer.upper()
                )

            # DB commit
            response_time = time.time() - start_time
            category = self._categorize_message(user_input)
            tokens_used = len(response["answer"].split())

            with next(self.db.get_session()) as session:
                interaction = ChatInteraction(
                    session_id=session_id,
                    user_message=user_input,
                    bot_response=answer,
                    response_time=response_time,
                    category=category,
                    tokens_used=tokens_used,
                    error_occurred=error_occurred,
                )
                session.add(interaction)
                session.commit()

            # Updating of the history with the answer
            self.conversation_history.append({"role": "assistant", "content": answer})

            print(f"Successful answer for the input: {user_input[:50]}...")
            return answer

        except Exception as e:
            error_occurred = 1
            error_msg = "Omlouvám se, ale při zpracování vaší otázky došlo k chybě."

            with next(self.db.get_session()) as session:
                interaction = ChatInteraction(
                    session_id=session_id,
                    user_message=user_input,
                    bot_response=error_msg,
                    response_time=time.time() - start_time,
                    category="error",
                    tokens_used=0,
                    error_occurred=error_occurred,
                )
                session.add(interaction)
                session.commit()

            print(f"Error while generating response: {str(e)}")
            return error_msg

    def clear_conversation(self) -> None:
        """Deleting the conversation history"""
        self.conversation_history.clear()
        print("Conversation history cleared")

    def get_conversation_history(self) -> List[Message]:
        """Returns the conversation history in the format required by the API"""
        return [
            Message(role=msg["role"], content=msg["content"])
            for msg in self.conversation_history
        ]


# Initialize chatbot
try:
    config = ChatbotConfig()
    chatbot = CoreChatbot(config)
except Exception as e:
    print(f"Failed to initialize chatbot: {str(e)}")
    raise


# API endpoints
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
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
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/clear", response_model=ClearResponse)
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
        print(f"Error in clear endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to clear conversation history"
        )


@app.get("/stats")
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


@app.get("/health")
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
            "version": app.version,
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
