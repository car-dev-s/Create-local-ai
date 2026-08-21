# Create-local-ai

A local RAG (Retrieval-Augmented Generation) stack: ingest your own documents, embed and store them in Chroma, and query them through a Streamlit UI backed by a local Ollama LLM.

Based on: [Build Your Local AI: From Zero to a Custom ChatGPT Interface with Ollama & Open WebUI](https://pahautelman.github.io/pahautelman-blog/tutorials/build-your-local-ai/build-your-local-ai/).

## Prerequisites

- Python (see `requirements.txt` for pinned package versions)
- [Ollama](https://ollama.com) running locally, with the models referenced in `config.yml` pulled:
  ```
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b
  ```

## Setup

```
pip install -r requirements.txt
```

Configuration lives in `config.yml` (data/index paths, embedding and chat models, chunking, prompts, search engine, logging). `config.py` loads it and resolves the `rag` directory paths relative to the project root.

If you use `search.py` (SerpApi web search), set `SERPAPI_API_KEY` in a `.env` file.

## Usage

1. Drop source documents into `resources/data_input/`.
2. Convert them to Markdown and build the Chroma index:
   ```
   python ingest.py
   ```
   Re-run this any time the files in `resources/data_input/` change; it rebuilds `resources/chroma_db` from scratch.
3. Ask questions about your documents:
   ```
   streamlit run rag_app.py
   ```

## Open WebUI (optional)

```
open-webui serve
```
Then open http://localhost:8080
