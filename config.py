"""Load configuration from config.yml."""
import logging
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yml"

with open(_CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

for _dir_key in ("data_input_dir", "data_markdown_dir", "chroma_dir"):
    config["rag"][_dir_key] = str(_PROJECT_ROOT / config["rag"][_dir_key])

_LOGGING_CONFIG = config.get("logging", {})

if "query_log" in _LOGGING_CONFIG:
    _LOGGING_CONFIG["query_log"]["file"] = str(_PROJECT_ROOT / _LOGGING_CONFIG["query_log"]["file"])
logging.basicConfig(
    level=_LOGGING_CONFIG.get("level", "INFO"),
    format=_LOGGING_CONFIG.get("format", "%(asctime)s %(levelname)s %(name)s: %(message)s"),
)
