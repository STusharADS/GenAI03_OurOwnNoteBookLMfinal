# NotebookLM-style RAG (Python)

A simple RAG pipeline and UI that lets users upload a PDF or TXT file, index it locally, and ask questions grounded in the document.

## Features
- Ingestion: PDF/TXT
- Chunking: RecursiveCharacterTextSplitter (documented, configurable)
- Embeddings: OpenRouter-compatible OpenAI embeddings
- Vector DB: Local Chroma (persistent on disk)
- Retrieval + grounded generation: LLM answers strictly from context
- Simple Streamlit UI

## Setup

1) Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

2) Set environment variables for OpenRouter:

```
export OPENROUTER_API_KEY="your_key_here"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

Optional tuning:

```
export EMBEDDING_MODEL="openai/text-embedding-3-large"
export CHAT_MODEL="openai/gpt-4.1-mini"
export CHROMA_DIR=".chroma"
export CHROMA_COLLECTION="SEC-B"
export CHUNK_SIZE="1000"
export CHUNK_OVERLAP="150"
export MAX_TOKENS="512"
```

3) Run the app:

```
streamlit run app.py
```

Notes:
- If you use a .env file, the app loads it automatically on startup.

## Chunking Strategy

Uses `RecursiveCharacterTextSplitter` with:
- Chunk size: 1000 characters
- Overlap: 150 characters

This keeps enough context while improving retrieval precision.

## Grounding Rule

The system prompt enforces:
- Answer only from retrieved context
- If the answer is not in context, say: "I do not know."

## Verification Checklist

- Upload a PDF and index it
- Ask 2-3 questions; verify answers are grounded in retrieved chunks
- Upload a different document and confirm answers change
