"""Convert files in resources/data_input to Markdown and index them into Chroma.

Run this whenever files under resources/data_input change:
    python ingest.py
"""
import logging
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from config import config

logger = logging.getLogger(__name__)

HEADER_PATTERN = re.compile(r"^## (.+)$", re.MULTILINE)


def split_by_title(content: str) -> list[tuple[str | None, str]]:
    """Split markdown content into (title, section_text) pairs at each '## ' header."""
    matches = list(HEADER_PATTERN.finditer(content))
    if not matches:
        return [(None, content)]

    sections = []
    if matches[0].start() > 0:
        sections.append((None, content[:matches[0].start()]))
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((title, content[start:end]))
    return sections

_RAG_CONFIG = config["rag"]
DATA_INPUT_DIR = Path(_RAG_CONFIG["data_input_dir"])
DATA_MARKDOWN_DIR = Path(_RAG_CONFIG["data_markdown_dir"])
CHROMA_DIR = _RAG_CONFIG["chroma_dir"]
EMBEDDING_MODEL = _RAG_CONFIG["embedding_model"]
CHUNK_SIZE = _RAG_CONFIG["chunk_size"]
CHUNK_OVERLAP = _RAG_CONFIG["chunk_overlap"]
EMBED_BATCH_SIZE = _RAG_CONFIG["embed_batch_size"]


def convert_to_markdown() -> list[Path]:
    DATA_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    markdown_paths = []

    for source_path in DATA_INPUT_DIR.iterdir():
        if not source_path.is_file():
            continue
        result = converter.convert(str(source_path))
        markdown_path = DATA_MARKDOWN_DIR / f"{source_path.stem}.md"
        markdown_path.write_text(result.text_content, encoding="utf-8")
        markdown_paths.append(markdown_path)
        logger.info("Converted %s -> %s", source_path.name, markdown_path.name)

    return markdown_paths


def build_vector_store(markdown_paths: list[Path]) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = []
    metadatas = []

    for markdown_path in markdown_paths:
        content = markdown_path.read_text(encoding="utf-8")
        for title, section_text in split_by_title(content):
            for chunk_index, chunk in enumerate(splitter.split_text(section_text)):
                texts.append(chunk)
                metadata = {"source": markdown_path.name}
                if title:
                    metadata["title"] = title
                    metadata["chunk_index"] = chunk_index
                metadatas.append(metadata)

    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[start:start + EMBED_BATCH_SIZE]
        batch_metadatas = metadatas[start:start + EMBED_BATCH_SIZE]
        vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)
        logger.info("Embedded %d/%d chunks", min(start + EMBED_BATCH_SIZE, len(texts)), len(texts))
    logger.info("Indexed %d chunks from %d file(s) into %s", len(texts), len(markdown_paths), CHROMA_DIR)


if __name__ == "__main__":
    paths = convert_to_markdown()
    if not paths:
        raise SystemExit(f"No files found in {DATA_INPUT_DIR}")
    build_vector_store(paths)
    logger.info("Done converting %d files to %s", len(paths), EMBEDDING_MODEL)
