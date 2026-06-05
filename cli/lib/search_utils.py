import json
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

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

GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_dataset.json"

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_K = 60
DEFAULT_ALPHA = 0.5
BM25_K1 = 1.5
BM25_B = 0.75

SCORE_PRECISION = 3
DOCUMENT_PREVIEW_LENGTH = 100

DEFAULT_CHUNK_SIZE = 200  # words
DEFAULT_CHUNK_OVERLAP = 40  # 20% of chunk size
DEFAULT_MAX_CHUNK_SIZE = 4  # sentences
DEFAULT_SEMANTIC_CHUNK_OVERLAP = 0
DEFAULT_RAG_CHUNK_OVERLAP = 1
DEFAULT_LIB_CHUNK_SEARCH_LIMIT = 10
DEFAULT_CLI_CHUNK_SEARCH_LIMIT = 5

SEARCH_EXPANSION_MULTIPLIER = 500
RERANK_RESULT_MULTIPLIER = 5
INDIVIDUAL_RERANK_DELAY_SECONDS = 3


class Movie(TypedDict):
    id: int
    title: str
    description: str


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class BM25Result(TypedDict):
    id: int
    score: float
    title: str
    description: str


class SemanticSearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]


class HybridSearchResult(TypedDict):
    id: int
    title: str
    description: str
    bm25_score: float
    semantic_score: float
    hybrid_score: float


class HybridRankResult(TypedDict):
    id: int
    title: str
    description: str
    bm25_rank: int | None
    semantic_rank: int | None
    rrf_score: float
    rerank_score: NotRequired[float]
    rerank_rank: NotRequired[int]
    rerank_cross_score: NotRequired[float]


class RRFSearchCommandResult(TypedDict):
    original_query: str
    enhanced_query: str | None
    enhance_method: Literal["spell", "rewrite", "expand"] | None
    query: str
    k: int
    rerank_method: Literal["individual", "batch", "cross_encoder"] | None
    reranked: bool
    results: list[HybridRankResult]


def load_movies() -> list[Movie]:
    with MOVIES_DATA_PATH.open("r", encoding="utf-8") as movies_file:
        data = json.load(movies_file)
    return cast(list[Movie], data["movies"])


def load_stopwords() -> list[str]:
    with STOPWORDS_DATA_PATH.open("r", encoding="utf-8") as stopwords_file:
        stopwords = stopwords_file.read()
    return stopwords.splitlines()


def format_search_result(
    doc_id: int,
    title: str,
    document: str,
    score: float,
    metadata: dict[str, Any] | None = None,
) -> SemanticSearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }
