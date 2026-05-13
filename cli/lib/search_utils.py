import json
from pathlib import Path
from typing import TypedDict, cast

DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOVIES_DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_DATA_PATH = PROJECT_ROOT / "data" / "stopwords.txt"

CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_DISK_DATA_PATH = CACHE_DIR / "index.pkl"
DOCMAP_DISK_DATA_PATH = CACHE_DIR / "docmap.pkl"


class Movie(TypedDict):
    id: int
    title: str
    description: str


def load_movies() -> list[Movie]:
    with MOVIES_DATA_PATH.open("r", encoding="utf-8") as movies_file:
        data = json.load(movies_file)
    return cast(list[Movie], data["movies"])


def load_stopwords() -> list[str]:
    with STOPWORDS_DATA_PATH.open("r", encoding="utf-8") as stopwords_file:
        stopwords = stopwords_file.read()
    return stopwords.splitlines()
