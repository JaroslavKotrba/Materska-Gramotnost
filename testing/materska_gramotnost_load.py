import os
import logging
from datetime import datetime
from typing import List
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
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

# Initialize OpenAI client
os.getcwd()
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load OpenAI embedding model to create vector store
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # default model
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# Load pdf
def load_pdf_document(file_path: str) -> List[Document]:
    pdf_reader = PdfReader(file_path)
    text_content = ""
    for page in pdf_reader.pages:
        text_content += page.extract_text() + "\n"

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    texts = text_splitter.split_text(text_content)

    return [
        Document(page_content=t, metadata={"source": os.path.basename(file_path)})
        for t in texts
    ]


# Load pdfs to create embeddings
def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    documents = []
    for file_path in file_paths:
        document = load_pdf_document(file_path)
        documents.extend(document)
    return documents


pdf_files = [  # add new documents here
    "../data/pdf/FS-slovnik-pojmu.pdf",
    "../data/pdf/Metodika-financni-gramotnost.pdf",
    "../data/pdf/MUNI-zaklady-financi.pdf",
]

documents = load_multiple_pdfs(pdf_files)

# Create FAISS vector store and retriever
# vector_store = FAISS.from_documents(documents, embeddings_model)

# SAVE
# vector_store.save_local("../data/vector_store")

# TESTING OF THE LOAD
vector_store = FAISS.load_local(
    "../data/vector_store", embeddings_model, allow_dangerous_deserialization=True
)

# Retrieve top 5 relevant chunks for the query
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Initialize chat model
chat_model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.4,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Create context-aware prompt template
prompt = ChatPromptTemplate.from_messages(
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

# Initialize global conversation history
conversation_history = []


def format_chat_history(history):
    return "\n".join(
        [
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in history[-4:]
        ]  # Include last 4 messages for context
    )


def get_response(user_input: str) -> str:
    """Generate a response to the user's input"""
    global conversation_history

    # Update conversation history with user input
    conversation_history.append({"role": "user", "content": user_input})

    # Create the document chain
    document_chain = create_stuff_documents_chain(
        llm=chat_model,
        prompt=prompt,
        document_variable_name="context",
    )

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    # Format chat history
    chat_history = format_chat_history(conversation_history)

    # Get response
    response = retrieval_chain.invoke(
        {
            "chat_history": chat_history,
            "input": user_input,
        }
    )

    # Extract the answer from the response
    answer = response["answer"]

    # Update conversation history with assistant's response
    conversation_history.append({"role": "assistant", "content": answer})

    return answer


def clear_conversation():
    """Clear the conversation history"""
    global conversation_history
    conversation_history.clear()


def main():
    try:
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
                clear_conversation()
                print("\nHistorie konverzace byla vymazána. Jak vám mohu pomoci?")
                continue
            elif user_input == "":
                continue

            response = get_response(user_input)
            print("\nAsistent:", response, "\n")

    except Exception as e:
        logger.critical(f"Critical error: {str(e)}")
        print(
            "\nOmlouváme se, ale došlo k neočekávané chybě. Prosím, zkuste aplikaci spustit znovu."
        )


if __name__ == "__main__":
    main()
