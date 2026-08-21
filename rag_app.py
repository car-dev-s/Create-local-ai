"""Streamlit UI for asking questions about the indexed documents.

Run with:
    streamlit run rag_app.py

Requires resources/chroma_db to already exist (run ingest.py first).
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import config

_RAG_CONFIG = config["rag"]
CHROMA_DIR = _RAG_CONFIG["chroma_dir"]
EMBEDDING_MODEL = _RAG_CONFIG["embedding_model"]
CHAT_MODEL = _RAG_CONFIG["chat_model"]
TOP_K = _RAG_CONFIG["top_k"]
PROMPT_TEMPLATE = _RAG_CONFIG["prompt_template"]

_QUERY_LOG_CONFIG = config["logging"]["query_log"]
query_logger = logging.getLogger("query_log")
query_logger.setLevel(logging.INFO)
query_logger.propagate = False
if not query_logger.handlers:
    Path(_QUERY_LOG_CONFIG["file"]).parent.mkdir(parents=True, exist_ok=True)
    _handler = RotatingFileHandler(
        _QUERY_LOG_CONFIG["file"],
        maxBytes=_QUERY_LOG_CONFIG["max_bytes"],
        backupCount=_QUERY_LOG_CONFIG["backup_count"],
        encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    query_logger.addHandler(_handler)


@st.cache_resource
def load_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


@st.cache_resource
def load_llm() -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL)


st.title("Local RAG")

if not Path(CHROMA_DIR).exists():
    st.error("No index found. Run `python ingest.py` first to build resources/chroma_db.")
    st.stop()

vector_store = load_vector_store()
llm = load_llm()

question = st.text_input("Ask a question about your documents")

if question:
    with st.spinner("Searching and generating an answer..."):
        docs = vector_store.similarity_search(question, k=TOP_K)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = llm.invoke(prompt)

    query_logger.info("Q: %s | A: %s", question, response.content)

    st.markdown("### Answer")
    st.write(response.content)

    st.markdown("### Sources")
    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    for source in sources:
        st.write(f"- {source}")
