# app.py
import os
import tempfile

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

import rag

st.set_page_config(page_title="NotebookLM RAG (Python)", page_icon=":page_with_curl:", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

    :root {
        --green-1: #0b3d2e;
        --green-2: #0f6b4d;
        --green-3: #19a06c;
        --green-4: #d8f5e6;
        --ink-1: #000000;
        --ink-2: #000000;
        --paper-1: #f4fbf7;
        --paper-2: #e7f5ee;
        --glow: rgba(25, 160, 108, 0.25);
    }

    html, body, [class*="stApp"] {
        background: radial-gradient(1200px 800px at 10% -10%, var(--paper-2), transparent 60%),
                    radial-gradient(900px 600px at 90% 10%, #d7efe3, transparent 55%),
                    var(--paper-1);
        color: var(--ink-1);
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
    }

    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        font-family: "Source Serif 4", "Georgia", serif;
        color: var(--ink-1);
        letter-spacing: 0.3px;
    }

    .stCaption {
        color: var(--ink-1);
        font-size: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--green-2), var(--green-3));
        color: white;
        border: 0;
        border-radius: 999px;
        padding: 0.6rem 1.3rem;
        font-weight: 600;
        box-shadow: 0 10px 24px var(--glow);
        transition: transform 120ms ease, box-shadow 120ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px var(--glow);
    }

    .stTextInput > div > div > input,
    .stFileUploader > div > div {
        border-radius: 14px;
        border: 1px solid rgba(15, 107, 77, 0.2);
        background: white;
        box-shadow: 0 6px 18px rgba(11, 61, 46, 0.06);
    }

    .stTextInput > div > div > input {
        background: #0b3d2e;
        color: #ffffff;
        border: 1px solid rgba(11, 61, 46, 0.6);
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.75);
    }

    .stFileUploader label,
    .stTextInput label,
    .stMarkdown label {
        color: var(--ink-1);
        font-weight: 600;
    }

    .stAlert {
        border-radius: 14px;
        border: 1px solid rgba(15, 107, 77, 0.15);
    }

    .stExpander {
        border-radius: 16px;
        border: 1px solid rgba(15, 107, 77, 0.18);
        background: white;
        box-shadow: 0 8px 20px rgba(11, 61, 46, 0.05);
    }

    .stDivider {
        border-color: rgba(15, 107, 77, 0.2);
    }

    .green-card {
        background: linear-gradient(135deg, #e8f7ef, #f6fffa);
        border: 1px solid rgba(15, 107, 77, 0.12);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 14px 30px rgba(11, 61, 46, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NotebookLM-style RAG (Python)")
st.caption(
    "Upload a PDF or TXT, index it, then ask questions grounded in the document. "
    "Uses Corrective RAG: retrieved chunks are graded for relevance and refined "
    "before the answer is generated."
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

st.markdown("<div class='green-card'>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload a document", type=["pdf", "txt"])

index_clicked = st.button("Index Document", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if index_clicked:
    if not uploaded:
        st.error("Please upload a PDF or TXT file first.")
    else:
        suffix = "." + uploaded.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Indexing document..."):
                vectorstore = rag.index_file(tmp_path, uploaded.name)
            st.session_state.vectorstore = vectorstore
            st.session_state.indexed_file = uploaded.name
            st.success(f"Indexed: {uploaded.name}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

st.divider()

st.subheader("Ask a question")
query = st.text_input("Your question", placeholder="e.g., Summarize the key ideas in this document.")
ask_clicked = st.button("Get Answer")

if ask_clicked:
    if not st.session_state.vectorstore:
        st.error("Please index a document first.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving, grading, refining, and generating answer..."):
            answer, chunks, info = rag.answer_query(
                query, st.session_state.vectorstore, k=3
            )

        st.markdown("### Answer")
        st.write(answer)

        action = info.get("action", "UNKNOWN")
        action_help = {
            "CORRECT": "All retrieved chunks were graded relevant.",
            "AMBIGUOUS": "Some retrieved chunks were irrelevant and dropped.",
            "INCORRECT": "No retrieved chunk was relevant to the question.",
        }
        st.markdown("### Corrective RAG")
        cols = st.columns(3)
        cols[0].metric("Action", action)
        cols[1].metric(
            "Relevant chunks",
            f"{info.get('relevant_count', 0)} / {info.get('retrieved_count', 0)}",
        )
        cols[2].metric("Refined", "Yes" if info.get("refined_knowledge") else "No")
        st.caption(action_help.get(action, ""))

        grades = info.get("grades", [])
        if grades:
            with st.expander("Relevance grading (per chunk)"):
                for g in grades:
                    mark = "✅" if g.get("relevant") else "❌"
                    reason = g.get("reason") or "—"
                    st.write(f"{mark} Chunk {g.get('index', 0) + 1}: {reason}")

        refined = info.get("refined_knowledge")
        if refined:
            with st.expander("Refined knowledge (sent to the LLM)"):
                st.write(refined)

        st.markdown("### Retrieved Chunks (Grounding)")
        relevant_idx = {g["index"] for g in grades if g.get("relevant")}
        for i, doc in enumerate(chunks, start=1):
            meta = doc.metadata or {}
            source = meta.get("source", st.session_state.indexed_file or "document")
            page = meta.get("page")
            status = "kept" if (i - 1) in relevant_idx else "dropped"
            header = (
                f"Chunk {i} [{status}] - {source}"
                + (f" (page {page})" if page is not None else "")
            )
            with st.expander(header):
                st.write(doc.page_content)