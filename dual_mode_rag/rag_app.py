"""Streamlit UI for asking questions about the indexed documents, routing between
passage-level lookup (vector search) and whole-document aggregate answers.

Run with either:
    streamlit run dual_mode_rag/rag_app.py
    python dual_mode_rag/run_app.py

Requires resources/chroma_db and resources/data_markdown to already exist (run ingest.py first).
"""
import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import config
from ingest import split_by_title

_RAG_CONFIG = config["rag"]
CHROMA_DIR = _RAG_CONFIG["chroma_dir"]
DATA_MARKDOWN_DIR = Path(_RAG_CONFIG["data_markdown_dir"])
EMBEDDING_MODEL = _RAG_CONFIG["embedding_model"]
CHAT_MODEL = _RAG_CONFIG["chat_model"]
TOP_K = _RAG_CONFIG["top_k"]
PROMPT_TEMPLATE = _RAG_CONFIG["prompt_template"]
AGGREGATE_KEYWORDS = _RAG_CONFIG["aggregate_keywords"]
AGGREGATE_PROMPT_TEMPLATE = _RAG_CONFIG["aggregate_prompt_template"]

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


def is_aggregate_query(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in AGGREGATE_KEYWORDS)


MENTION_COUNT_PATTERN = re.compile(
    r"how many times\s+(?:is\s+|does\s+|do\s+)?([a-zA-Z][\w'-]*)\s*(?:is\s+|are\s+)?(?:mentioned|appears?)",
    re.IGNORECASE,
)

WHERE_MENTIONED_PATTERN = re.compile(
    r"where\s+(?:is\s+|does\s+|do\s+)?([a-zA-Z][\w'-]*)\s*(?:is\s+|are\s+)?(?:mentioned|appears?)",
    re.IGNORECASE,
)


def _count_mentions_by_title(entity: str) -> dict[str, int]:
    """Count literal occurrences of entity per '## TITLE' section across the indexed markdown files."""
    pattern = re.compile(rf"\b{re.escape(entity)}\b", re.IGNORECASE)
    counts_by_title: dict[str, int] = {}
    for path in sorted(DATA_MARKDOWN_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        for title, section_text in split_by_title(content):
            count = len(pattern.findall(section_text))
            if count:
                key = title or path.name
                counts_by_title[key] = counts_by_title.get(key, 0) + count
    return counts_by_title


def answer_mention_count(question: str) -> tuple[str, list[str], list[str]] | None:
    """Count literal occurrences of a named entity in the source files, without asking the LLM to count."""
    match = MENTION_COUNT_PATTERN.search(question)
    if not match:
        return None

    entity = match.group(1)
    counts_by_title = _count_mentions_by_title(entity)
    total = sum(counts_by_title.values())
    sources = [path.name for path in sorted(DATA_MARKDOWN_DIR.glob("*.md"))]
    if counts_by_title:
        breakdown = "; ".join(
            f"{title} ({count})" for title, count in sorted(counts_by_title.items(), key=lambda item: -item[1])
        )
        answer = f"'{entity}' is mentioned {total} time(s) across the indexed document(s). By story: {breakdown}."
    else:
        answer = f"'{entity}' is mentioned {total} time(s) across the indexed document(s)."
    return answer, sources, []


def answer_where_mentioned(question: str) -> tuple[str, list[str], list[str]] | None:
    """List which '## TITLE' sections mention a named entity, without asking the LLM to locate it."""
    match = WHERE_MENTIONED_PATTERN.search(question)
    if not match:
        return None

    entity = match.group(1)
    counts_by_title = _count_mentions_by_title(entity)
    sources = [path.name for path in sorted(DATA_MARKDOWN_DIR.glob("*.md"))]
    if counts_by_title:
        listing = "; ".join(
            f"{title} ({count})" for title, count in sorted(counts_by_title.items(), key=lambda item: -item[1])
        )
        answer = f"'{entity}' is mentioned in: {listing}."
    else:
        answer = f"'{entity}' was not found in the indexed document(s)."
    return answer, sources, []


def answer_lookup(vector_store: Chroma, llm: ChatOllama, question: str) -> tuple[str | list[str | Any], list[Any], list[str]]:
    docs = vector_store.similarity_search(question, k=TOP_K)
    if docs:
        top_title = docs[0].metadata.get("title")
        if top_title:
            result = vector_store.get(where={"title": top_title})
            docs = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in sorted(
                    zip(result["documents"], result["metadatas"]),
                    key=lambda item: item[1].get("chunk_index", 0),
                )
            ]
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    passages = [doc.page_content for doc in docs]
    return response.content, sources, passages


def answer_aggregate(llm: ChatOllama, question: str) -> tuple[str | list[str | Any], list[str], list[str]]:
    markdown_paths = sorted(DATA_MARKDOWN_DIR.glob("*.md"))
    context = "\n\n".join(
        f"# {path.name}\n{path.read_text(encoding='utf-8')}" for path in markdown_paths
    )
    prompt = AGGREGATE_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    sources = [path.name for path in markdown_paths]
    return response.content, sources, []


def main() -> None:
    st.title("Local RAG (dual-mode)")

    if not Path(CHROMA_DIR).exists():
        st.error("No index found. Run `python ingest.py` first to build resources/chroma_db.")
        st.stop()

    vector_store = load_vector_store()
    llm = load_llm()

    question = st.text_input("Ask a question about your documents")

    if question:
        mention_count = answer_mention_count(question)
        where_mentioned = answer_where_mentioned(question) if mention_count is None else None
        aggregate = is_aggregate_query(question)
        if mention_count is not None:
            mode_label = "mention count"
        elif where_mentioned is not None:
            mode_label = "mention location"
        else:
            mode_label = "whole-document" if aggregate else "passage lookup"
        with st.spinner(f"Answering ({mode_label} mode)..."):
            if mention_count is not None:
                answer, sources, passages = mention_count
            elif where_mentioned is not None:
                answer, sources, passages = where_mentioned
            elif aggregate:
                answer, sources, passages = answer_aggregate(llm, question)
            else:
                answer, sources, passages = answer_lookup(vector_store, llm, question)

        query_logger.info("[%s] Q: %s | A: %s", mode_label, question, answer)

        st.caption(f"Mode: {mode_label}")
        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        for source in sources:
            st.write(f"- {source}")

        if passages:
            with st.expander(f"Show {len(passages)} raw source passage(s) used to generate this answer"):
                for i, passage in enumerate(passages, start=1):
                    st.markdown(f"**Passage {i}**")
                    st.text(passage)


if __name__ == "__main__":
    main()
