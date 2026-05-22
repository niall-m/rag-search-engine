import json
from pathlib import Path
from typing import TypedDict, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOVIES_DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_DATA_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
CACHE_DIR = PROJECT_ROOT / "cache"
INDEX_DISK_DATA_PATH = CACHE_DIR / "index.pkl"
DOCMAP_DISK_DATA_PATH = CACHE_DIR / "docmap.pkl"
TF_DISK_DATA_PATH = CACHE_DIR / "term_frequencies.pkl"
DOC_LENGTHS_PATH = CACHE_DIR / "doc_lengths.pkl"

MOVIE_EMBEDDINGS_PATH = CACHE_DIR / "movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = CACHE_DIR / "chunk_embeddings.npy"
CHUNK_METADATA_PATH = CACHE_DIR / "chunk_metadata.json"

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_CHUNK_SIZE = 200  # words
DEFAULT_CHUNK_OVERLAP = 40  # 20% of chunk size
DEFAULT_MAX_CHUNK_SIZE = 4  # sentences
DEFAULT_SEMANTIC_CHUNK_OVERLAP = 0


class Movie(TypedDict):
    id: int
    title: str
    description: str


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


def load_movies() -> list[Movie]:
    with MOVIES_DATA_PATH.open("r", encoding="utf-8") as movies_file:
        data = json.load(movies_file)
    return cast(list[Movie], data["movies"])


def load_stopwords() -> list[str]:
    with STOPWORDS_DATA_PATH.open("r", encoding="utf-8") as stopwords_file:
        stopwords = stopwords_file.read()
    return stopwords.splitlines()
