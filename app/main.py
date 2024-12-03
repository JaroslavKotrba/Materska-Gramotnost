# TO RUN APP

# conda env create -f environment.yml
# conda create -n mat_gram python=3.12
# conda env remove -n mat_gram

# uvicorn app.main:app --reload

# conda env export --name mat_gram > environment.yml
# pip list --format=freeze > requirements.txt

# conda env export --name mat_gram > environment.yml
# pip list --format=freeze > requirements.txt

import os
from typing import List, Dict
from datetime import datetime
import time
import psutil
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
from dotenv import load_dotenv

# Path
os.getcwd()

# FastAPI app initialization
app = FastAPI(
    title="Financial Chatbot API",
    description="API for a Czech maternity advisory chatbot",
    version="1.0.0",
)

# Initialize start time for uptime tracking
START_TIME = time.time()

# CLIENT DOMAIN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://jaroslavkotrba.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static files (CSS, JavaScript)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize Jinja2 for templates - index.html
templates = Jinja2Templates(directory="app/templates")


# Pydantic models for request/response validation
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    conversation_history: List[Message]


class ClearResponse(BaseModel):
    message: str
    status: bool


# Model
class ChatbotConfig:
    """Class for chatbot configuration"""

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not available in .env")

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
                    """Jste přátelský český chatbot zaměřený na podporu a poradenství pro maminky s miminky.
                
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
                
                U zdravotních témat vždy upozorněte, že je důležité konzultovat konkrétní situaci s lékařem
                nebo specializovaným odborníkem (pediatr, laktační poradkyně, fyzioterapeut apod.).
                
                Doporučené zdroje pro ověření informací:
                - www.kojeni.cz pro informace o kojení
                - www.pediatrie.cz pro zdravotní informace
                - www.mpsv.cz pro informace o rodičovském příspěvku
                
                Pokud si nejste něčím jistí nebo je otázka mimo vaši oblast, přiznejte to a doporučte
                konzultaci s příslušným odborníkem.
                
                Na konci odpovědi se vždy slušně zeptejte, zda můžete pomoci s něčím dalším.
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

    def format_chat_history(self) -> str:
        """Format the conversation history for context inclusion.

        Returns a string of the last N interactions based on max_history setting."""
        return "\n".join(
            [
                f"{'Uživatel' if msg['role'] == 'user' else 'Asistent'}: {msg['content']}"
                for msg in self.conversation_history[-self.config.max_history :]
            ]
        )

    def get_response(self, user_input: str) -> str:
        """Generate a response to user input using the language model.

        Updates conversation history and handles any errors during processing.
        Returns an error message if response generation fails."""
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

            # Updating of the history with the answer
            self.conversation_history.append({"role": "assistant", "content": answer})

            print(f"Successful answer for the input: {user_input[:50]}...")
            return answer

        except Exception as e:
            error_msg = "Omlouvám se, ale při zpracování vaší otázky došlo k chybě."
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
        response = chatbot.get_response(request.message)
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
