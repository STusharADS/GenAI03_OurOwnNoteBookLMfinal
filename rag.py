# rag.py
import os
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4.1-mini")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "SEC-B")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

SYSTEM_PROMPT = (
    "You are an AI assistant. Answer only using the provided context. "
    "If the answer is not in the context, say you do not know. "
    "Cite the document content when possible."
)

def _load_documents(file_path: str, filename: str) -> List[Document]:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file type. Only PDF or TXT is allowed.")
    return loader.load()

def _chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def _openai_kwargs() -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    kwargs = {}
    if api_key:
        kwargs["openai_api_key"] = api_key
    if base_url:
        kwargs["openai_api_base"] = base_url
    return kwargs

def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, **_openai_kwargs())

def _get_vectorstore(embeddings, collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

def index_file(file_path: str, filename: str):
    docs = _load_documents(file_path, filename)
    chunks = _chunk_documents(docs)
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings, CHROMA_COLLECTION)
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    return vectorstore

def answer_query(query: str, vectorstore, k: int = 3) -> Tuple[str, List[Document]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chunks = retriever.invoke(query)

    context_text = "\n\n".join(
        [f"[Chunk {i+1}] {doc.page_content}" for i, doc in enumerate(chunks)]
    )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}"
    )

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=MAX_TOKENS,
        **_openai_kwargs(),
    )
    response = llm.invoke(prompt)
    return response.content, chunks