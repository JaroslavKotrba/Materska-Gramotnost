import os
from typing import List
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

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
    "../data/pdf/kniha_pro_rodice.pdf",
    "../data/pdf/psychomotoricky_vyvoj_ditete.pdf",
    "../data/pdf/vyziva_deti.pdf",
]

documents = load_multiple_pdfs(pdf_files)

# Create FAISS vector store
# vector_store = FAISS.from_documents(documents, embeddings_model)

# Save FAISS vector store
# vector_store.save_local("../data/vector_store")

# TESTING OF THE LOAD
vector_store = FAISS.load_local(
    "../data/vector_store", embeddings_model, allow_dangerous_deserialization=True
)
