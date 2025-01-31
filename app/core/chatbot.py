import os
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
from ..database.models import ChatInteraction
from ..schemas.models import Message
from dotenv import load_dotenv


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
