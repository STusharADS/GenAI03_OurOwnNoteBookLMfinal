# rag.py
import json
import os
import re
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

# Corrective RAG: an LLM judges whether each retrieved chunk actually helps
# answer the question. Irrelevant chunks are dropped before generation.
GRADER_PROMPT = (
    "You are a strict retrieval grader for a document QA system. "
    "For each numbered chunk, decide whether it contains information that "
    "helps answer the user's question. Judge relevance only - do not answer "
    "the question.\n\n"
    "Return ONLY a JSON object of this exact shape, no prose, no markdown:\n"
    '{{"grades": [{{"chunk": 1, "relevant": true, "reason": "short reason"}}]}}\n\n'
    "Question: {question}\n\n"
    "Chunks:\n{chunks}"
)

# Knowledge refinement (decompose-then-recompose): keep only the sentences /
# strips from the relevant chunks that bear on the question, drop the rest.
REFINE_PROMPT = (
    "Extract only the sentences or passages from the context below that are "
    "directly useful for answering the question. Remove anything unrelated, "
    "redundant, or boilerplate. Preserve wording from the source - do not "
    "summarize, infer, or add information. If nothing in the context is "
    "useful, reply with exactly: NONE\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}"
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


def _get_llm(max_tokens: int = MAX_TOKENS) -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=max_tokens,
        **_openai_kwargs(),
    )


def index_file(file_path: str, filename: str):
    docs = _load_documents(file_path, filename)
    chunks = _chunk_documents(docs)
    embeddings = _get_embeddings()
    vectorstore = _get_vectorstore(embeddings, CHROMA_COLLECTION)
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    return vectorstore


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response, tolerating fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def _grade_documents(query: str, chunks: List[Document], llm: ChatOpenAI) -> List[dict]:
    """Grade each chunk for relevance. Returns one dict per chunk:
    {"index", "relevant", "reason"}. On grader failure, fail open (keep all
    chunks) so CRAG degrades to plain RAG rather than breaking."""
    if not chunks:
        return []

    numbered = "\n\n".join(
        f"[Chunk {i + 1}]\n{doc.page_content}" for i, doc in enumerate(chunks)
    )
    prompt = GRADER_PROMPT.format(question=query, chunks=numbered)

    try:
        raw = llm.invoke(prompt).content
        parsed = _extract_json(raw)
        by_index = {
            int(g["chunk"]) - 1: g for g in parsed.get("grades", []) if "chunk" in g
        }
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        by_index = {}

    grades = []
    for i in range(len(chunks)):
        g = by_index.get(i)
        if g is None:
            # Missing/unparseable grade -> keep the chunk (fail open).
            grades.append(
                {"index": i, "relevant": True, "reason": "grader unavailable"}
            )
        else:
            grades.append(
                {
                    "index": i,
                    "relevant": bool(g.get("relevant", True)),
                    "reason": str(g.get("reason", "")).strip(),
                }
            )
    return grades


def _refine_knowledge(
    query: str, relevant_docs: List[Document], llm: ChatOpenAI
) -> str:
    """Decompose-then-recompose: strip relevant docs down to the passages that
    actually bear on the question. Falls back to raw concatenation on failure."""
    if not relevant_docs:
        return ""

    raw_context = "\n\n".join(
        f"[Chunk {i + 1}] {doc.page_content}"
        for i, doc in enumerate(relevant_docs)
    )
    prompt = REFINE_PROMPT.format(question=query, context=raw_context)

    try:
        refined = llm.invoke(prompt).content.strip()
    except Exception:
        return raw_context

    if not refined or refined.strip().upper() == "NONE":
        return ""
    return refined


def answer_query(
    query: str, vectorstore, k: int = 3
) -> Tuple[str, List[Document], dict]:
    """Corrective RAG: retrieve -> grade -> refine -> generate.

    Returns (answer, retrieved_chunks, info) where info carries the CRAG
    decision trace: per-chunk grades, the action taken, and refined knowledge.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chunks = retriever.invoke(query)

    info = {
        "action": "INCORRECT",
        "grades": [],
        "relevant_count": 0,
        "retrieved_count": len(chunks),
        "refined_knowledge": "",
    }

    if not chunks:
        return (
            "I could not find anything in this document related to your "
            "question.",
            chunks,
            info,
        )

    llm = _get_llm()

    grades = _grade_documents(query, chunks, llm)
    info["grades"] = grades

    relevant_docs = [chunks[g["index"]] for g in grades if g["relevant"]]
    relevant_count = len(relevant_docs)
    info["relevant_count"] = relevant_count

    # CRAG action selection based on how much retrieved knowledge survived
    # the relevance check.
    if relevant_count == 0:
        info["action"] = "INCORRECT"
        return (
            "The document does not appear to contain information needed to "
            "answer this question, so I can't answer it from this document.",
            chunks,
            info,
        )
    elif relevant_count == len(chunks):
        info["action"] = "CORRECT"
    else:
        info["action"] = "AMBIGUOUS"

    refined_knowledge = _refine_knowledge(query, relevant_docs, llm)
    info["refined_knowledge"] = refined_knowledge

    if not refined_knowledge:
        info["action"] = "INCORRECT"
        return (
            "The document does not appear to contain information needed to "
            "answer this question, so I can't answer it from this document.",
            chunks,
            info,
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{refined_knowledge}\n\n"
        f"Question: {query}"
    )
    response = llm.invoke(prompt)
    return response.content, chunks, info
