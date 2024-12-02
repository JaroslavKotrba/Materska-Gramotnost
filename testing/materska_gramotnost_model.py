import os
import logging
from typing import List, Dict
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Configure logging to track chatbot operations and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f'logs/chatbot_{datetime.now().strftime("%Y%m%d")}.log',
)
logger = logging.getLogger(__name__)


class ChatbotConfig:
    """Configuration class that manages all chatbot parameters and settings.

    Handles environment variables, model settings, and conversation parameters.
    Raises ValueError if required API keys are missing."""

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
                "../data/vector_store",
                self.embeddings_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.error(f"Error while loading of the vector store: {str(e)}")
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

            answer = response["answer"]

            # Updating of the history with the answer
            self.conversation_history.append({"role": "assistant", "content": answer})

            logger.info(f"Successful answer for the input: {user_input[:50]}...")
            return answer

        except Exception as e:
            error_msg = "Omlouvám se, ale při zpracování vaší otázky došlo k chybě."
            logger.error(f"Error while generating response: {str(e)}")
            return error_msg

    def clear_conversation(self) -> None:
        """Reset the conversation history.

        Clears all stored messages and logs the action."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")


def main():
    """Main application entry point.

    Initializes the chatbot and handles the main conversation loop.
    Provides basic command processing for quitting and clearing history."""
    try:
        config = ChatbotConfig()
        chatbot = CoreChatbot(config)

        print("\n=== Virtuální pomocník pro maminky ===")
        print("Pro ukončení napište 'quit'")
        print("Pro vymazání historie napište 'clear'")
        print("Jak vám mohu dnes pomoci s péčí o miminko?\n")

        while True:
            user_input = input("Vy: ").strip()

            if user_input.lower() == "quit":
                print("\nNashledanou! Přeji hezký den!")
                break
            elif user_input.lower() == "clear":
                chatbot.clear_conversation()
                print("\nHistorie konverzace byla vymazána. Jak vám mohu pomoci?")
                continue
            elif user_input == "":
                continue

            response = chatbot.get_response(user_input)
            print("\nAsistent:", response, "\n")

    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        print(
            "\nOmlouváme se, ale došlo k neočekávané chybě. Prosím, zkuste aplikaci spustit znovu."
        )


if __name__ == "__main__":
    main()
