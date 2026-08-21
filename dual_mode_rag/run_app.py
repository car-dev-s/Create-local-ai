"""Launches the Streamlit RAG app as a subprocess.

Run with:
    python dual_mode_rag/run_app.py

Stopping this process (Ctrl+C) also stops the Streamlit server.
"""
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "rag_app.py"


def main() -> None:
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(APP_PATH)])
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
