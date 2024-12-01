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

# Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f'logs/chatbot_{datetime.now().strftime("%Y%m%d")}.log',
)
logger = logging.getLogger(__name__)


# Model
class ChatbotConfig:
    """Class for chatbot configuration"""

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not available in .env")

        self.model_name = "gpt-4o-mini"
        self.temperature = 0.4
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.top_k_results = 5
        self.max_history = 4


class FinancialChatbot:
    """Main class of the chatbot"""

    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.conversation_history: List[Dict[str, str]] = []

        # Init of the embedding model
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=config.openai_api_key
        )

        self.chat_model = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key,
        )

        # Load of the vectore store
        self.vector_store = self._load_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.top_k_results}
        )

        # Creation of the prompt template
        self.prompt = self._create_prompt_template()

        # Init of chains
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
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Jste přátelský český chatbot zaměřený na finanční poradenství a finanční gramotnost.
                Používejte následující kontext k odpovědím na otázky o financích. Odpovědi piště vykáním a spisovnou češtinou.
                Dále by odpovědi měly být jasné, srozumitelné a vedené v přátelském tónu. Pokud si nejste něčím jistí, přiznejte to 
                a navrhněte, kde mohou uživatelé najít více informací. Pokud je otázka mimo finanční oblast, zdvořile to vysvětlete.""",
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
        """Formatting of the conversation history"""
        return "\n".join(
            [
                f"{'Uživatel' if msg['role'] == 'user' else 'Asistent'}: {msg['content']}"
                for msg in self.conversation_history[-self.config.max_history :]
            ]
        )

    def get_response(self, user_input: str) -> str:
        """Generating of the answear for the user's input"""
        try:
            # Updating of the history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Getting the answear
            response = self.retrieval_chain.invoke(
                {
                    "chat_history": self.format_chat_history(),
                    "input": user_input,
                }
            )

            answer = response["answer"]

            # Updating of the history with the answear
            self.conversation_history.append({"role": "assistant", "content": answer})

            logger.info(f"Successful answear for the input: {user_input[:50]}...")
            return answer

        except Exception as e:
            error_msg = "Omlouvám se, ale při zpracování vaší otázky došlo k chybě."
            logger.error(f"Error while generating response: {str(e)}")
            return error_msg

    def clear_conversation(self) -> None:
        """Deleting the conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")


def main():
    try:
        config = ChatbotConfig()
        chatbot = FinancialChatbot(config)

        print("\n=== Chatbot finančního poradce ===")
        print("Pro ukončení napište 'quit'")
        print("Pro vymazání historie napište 'clear'")
        print("Jak vám mohu dnes pomoci s vašimi finančními otázkami?\n")

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
