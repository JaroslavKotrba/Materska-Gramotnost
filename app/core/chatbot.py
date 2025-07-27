import os
import logging
import time
import uuid
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from ..database import db
from ..schemas.models import ChatInteraction
from ..schemas.models import Message
from dotenv import load_dotenv

# Logger
logger = logging.getLogger(__name__)


class ChatbotConfig:
    """Class for chatbot configuration"""

    def __init__(self):
        logger.info("Initializing ChatbotConfig")
        load_dotenv()

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY is not available in .env")
            raise ValueError("OPENAI_API_KEY is not available in .env")
        else:
            logger.debug("OPENAI_API_KEY loaded successfully")

        self.admin_api_key = os.getenv("ADMIN_API_KEY")
        if not self.admin_api_key:
            logger.error("ADMIN_API_KEY is not available in .env")
            raise ValueError("ADMIN_API_KEY is not available in .env")
        else:
            logger.debug("ADMIN_API_KEY loaded successfully")

        self.model_name = "gpt-4o-mini"
        self.temperature = 0.4  # Controls randomness in responses
        self.chunk_size = 1000  # Size of text chunks for processing
        self.chunk_overlap = 200  # Overlap between chunks to maintain context
        self.top_k_results = 5  # Number of similar chunks to retrieve
        self.max_history = 4  # Maximum number of conversation turns to remember

        logger.info(
            f"Configuration loaded - Model: {self.model_name}, Temperature: {self.temperature}, "
            f"Top K: {self.top_k_results}, Max History: {self.max_history}"
        )


class CoreChatbot:
    """Core chatbot implementation.

    Integrates OpenAI's language models with FAISS vector storage
    for context-aware responses"""

    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []
        self.db = db
        self.current_session_id = str(uuid.uuid4())

        logger.info(
            f"Initializing CoreChatbot with session_id: {self.current_session_id}"
        )
        logger.info("Loading models and initializing components...")

        # Initialize embedding model for text vectorization
        try:
            self.embeddings_model = OpenAIEmbeddings(
                model="text-embedding-ada-002", openai_api_key=config.openai_api_key
            )
            logger.info("Successfully loaded embedding model (text-embedding-ada-002)")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

        # Set up the main language model for generating responses
        try:
            self.chat_model = ChatOpenAI(
                model_name=config.model_name,
                temperature=config.temperature,
                openai_api_key=config.openai_api_key,
            )
            logger.info(f"Successfully loaded chat model: {config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load chat model {config.model_name}: {str(e)}")
            raise

        # Initialize vector store and retrieval system
        try:
            self.vector_store = self._load_vector_store()
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": config.top_k_results}
            )
            logger.info(
                f"Successfully initialized vector store and retriever (top_k={config.top_k_results})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store or retriever: {str(e)}")
            raise

        # Set up conversation template and processing chains
        try:
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
            logger.info("Successfully created prompt template and processing chains")
        except Exception as e:
            logger.error(
                f"Failed to create prompt template or processing chains: {str(e)}"
            )
            raise

        logger.info("CoreChatbot initialization completed successfully")

    def _load_vector_store(self) -> FAISS:
        """Load the FAISS vector store from disk.

        Contains pre-processed knowledge base.
        Raises Exception if loading fails."""
        logger.debug("Attempting to load FAISS vector store from data/vector_store")
        try:
            vector_store = FAISS.load_local(
                "data/vector_store",
                self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
            return vector_store
        except Exception as e:
            logger.error(f"Error while loading the vector store: {str(e)}")
            logger.error(f"Vector store path: data/vector_store")
            raise

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the conversation prompt template.

        Defines the chatbot's personality, expertise areas, and response guidelines
        for providing advice."""
        logger.debug("Creating prompt template for conversation")
        try:
            prompt = ChatPromptTemplate.from_messages(
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
                    Odpovědi by měly být stručné, věcné, srozumitelné a vedené v přátelském, podporujícím tónu.
                    
                    Pokud si nejste něčím jistí nebo je otázka mimo vaši oblast, přiznejte to a pokuste se i tak stručně odpovědět.
                    
                    Neopakujte zadanou otázku, vynechte zbytečné fráze a dlouhé nabídky další pomoci.
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
            logger.debug("Prompt template created successfully")
            return prompt
        except Exception as e:
            logger.error(f"Failed to create prompt template: {str(e)}")
            raise

    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritical marks from Czech"""
        logger.debug(f"Removing diacritics from text: {text[:50]}...")
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
        result = "".join(replacements.get(c.lower(), c.lower()) for c in text)
        logger.debug(f"Diacritics removed, result: {result[:50]}...")
        return result

    def _categorize_message(self, message: str) -> str:
        """Categorize the incoming message based on content"""
        logger.debug(f"Categorizing message: {message[:50]}...")
        categories = {
            "pece o miminko": ["miminko", "kojeni", "kojit", "spat", "vyvoj", "dite"],
            "pece o maminku": ["maminka", "porodni", "cviceni", "deprese"],
            "prakticke rady": ["vybavicka", "kocarek", "nositko", "prikrm"],
            "socialni podpora": ["materska", "rodicovska", "prispevek", "davky"],
        }

        message_normalized = self._remove_diacritics(message).lower()
        for category, keywords in categories.items():
            if any(keyword in message_normalized for keyword in keywords):
                logger.debug(f"Message categorized as: {category}")
                return category

        logger.debug("Message categorized as: ostatni")
        return "ostatni"

    def format_chat_history(self) -> str:
        """Format the conversation history for context inclusion.

        Returns a string of the last N interactions based on max_history setting."""
        history_length = len(self.conversation_history)
        max_history = self.config.max_history
        logger.debug(
            f"Formatting chat history: {history_length} messages, showing last {max_history}"
        )

        formatted_history = "\n".join(
            [
                f"{'Uživatel' if msg['role'] == 'user' else 'Asistent'}: {msg['content']}"
                for msg in self.conversation_history[-max_history:]
            ]
        )
        logger.debug(f"Formatted history length: {len(formatted_history)} characters")
        return formatted_history

    def get_response(self, user_input: str, session_id: str = "") -> str:
        """Generate a response to user input using the language model.

        Updates conversation history and handles any errors during processing.
        Returns an error message if response generation fails."""

        # DB values
        session_id = session_id or self.current_session_id
        start_time = time.time()
        error_occurred = 0
        tokens_used = 0

        logger.info(
            f"Processing user input - Session: {session_id}, Input: '{user_input[:100]}'"
        )

        try:
            # Updating history
            self.conversation_history.append({"role": "user", "content": user_input})
            logger.debug(
                f"Added user message to conversation history. Total messages: {len(self.conversation_history)}"
            )

            # Getting answer
            logger.debug("Invoking retrieval chain...")
            chain_start_time = time.time()
            response = self.retrieval_chain.invoke(
                {
                    "chat_history": self.format_chat_history(),
                    "input": user_input,
                }
            )
            chain_time = time.time() - chain_start_time
            logger.debug(f"Retrieval chain completed in {chain_time:.2f}s")

            # Remove asterisks from the response
            answer = response["answer"].replace("*", "")
            logger.debug(
                f"Raw response length: {len(response['answer'])}, Cleaned length: {len(answer)}"
            )

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

            logger.info(
                f"Response generated successfully - Category: {category}, "
                f"Response time: {response_time:.2f}s, Tokens: {tokens_used}"
            )

            try:
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
                logger.debug("Interaction saved to database successfully")
            except Exception as db_error:
                logger.error(f"Failed to save interaction to database: {str(db_error)}")
                # Don't fail the entire request if DB save fails

            # Updating of the history with the answer
            self.conversation_history.append({"role": "assistant", "content": answer})
            logger.debug(
                f"Added assistant response to conversation history. Total messages: {len(self.conversation_history)}"
            )

            logger.info(
                f"Successful response for input: '{user_input[:100]}' -> '{answer[:50]}...'"
            )
            return answer

        except Exception as e:
            error_occurred = 1
            error_msg = "Omlouvám se, ale při zpracování vaší otázky došlo k chybě."
            response_time = time.time() - start_time

            logger.error(
                f"Error generating response for input '{user_input[:50]}...': {str(e)}"
            )
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Response time before error: {response_time:.2f}s")

            try:
                with next(self.db.get_session()) as session:
                    interaction = ChatInteraction(
                        session_id=session_id,
                        user_message=user_input,
                        bot_response=error_msg,
                        response_time=response_time,
                        category="error",
                        tokens_used=0,
                        error_occurred=error_occurred,
                    )
                    session.add(interaction)
                    session.commit()
                logger.debug("Error interaction saved to database")
            except Exception as db_error:
                logger.error(
                    f"Failed to save error interaction to database: {str(db_error)}"
                )

            return error_msg

    def clear_conversation(self) -> None:
        """Deleting the conversation history"""
        history_length = len(self.conversation_history)
        self.conversation_history.clear()
        logger.info(f"Conversation history cleared - Removed {history_length} messages")

    def get_conversation_history(self) -> List[Message]:
        """Returns the conversation history in the format required by the API"""
        history_length = len(self.conversation_history)
        logger.debug(f"Retrieving conversation history - {history_length} messages")
        return [
            Message(role=msg["role"], content=msg["content"])
            for msg in self.conversation_history
        ]
