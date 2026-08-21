import logging
import os

from dotenv import load_dotenv
from serpapi import GoogleSearch

from config import config

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("SERPAPI_API_KEY")
ENGINE = config["search"]["engine"]


def search(query: str) -> dict:
    params = {
        "engine": ENGINE,
        "q": query,
        "api_key": API_KEY,
    }
    return GoogleSearch(params).get_dict()


if __name__ == "__main__":
    results = search("python serpapi")
    for result in results.get("organic_results", []):
        logger.info("%s - %s", result.get("title"), result.get("link"))
