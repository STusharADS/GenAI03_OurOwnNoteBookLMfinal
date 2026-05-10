# app.py
import os
import tempfile

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

import rag

st.set_page_config(page_title="NotebookLM RAG (Python)", page_icon=":page_with_curl:", layout="wide")

st.title("NotebookLM-style RAG (Python)")
st.caption("Upload a PDF or TXT, index it, then ask questions grounded in the document.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

uploaded = st.file_uploader("Upload a document", type=["pdf", "txt"])

index_clicked = st.button("Index Document", use_container_width=True)

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
        with st.spinner("Retrieving and generating answer..."):
            answer, chunks = rag.answer_query(query, st.session_state.vectorstore, k=3)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Retrieved Chunks (Grounding)")
        for i, doc in enumerate(chunks, start=1):
            meta = doc.metadata or {}
            source = meta.get("source", st.session_state.indexed_file or "document")
            page = meta.get("page")
            header = f"Chunk {i} - {source}" + (f" (page {page})" if page is not None else "")
            with st.expander(header):
                st.write(doc.page_content)