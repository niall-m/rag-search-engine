import json
import re
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
    MOVIE_EMBEDDINGS_PATH,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_LIB_CHUNK_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_OVERLAP,
    DEFAULT_RAG_CHUNK_OVERLAP,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DOCUMENT_PREVIEW_LENGTH,
    ChunkMetadata,
    Movie,
    load_movies,
    SearchResult,
    format_search_result,
)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("text must contain actual text")
        return self.model.encode([text])[0]

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities: list[tuple[float, Movie]] = []
        for document, embedding in zip(self.documents, self.embeddings):
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((similarity, document))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, document in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": document["title"],
                    "description": document["description"],
                }
            )

        return results


def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_search(query, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"  {result['description']}")
        print()


def create_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    step = chunk_size - overlap
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = create_chunks(text, chunk_size, overlap)

    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def create_semantic_chunks(
    text: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_SEMANTIC_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("overlap must be between 0 and max_chunk_size - 1")

    step = max_chunk_size - overlap
    i = 0
    while i < len(sentences):
        chunks.append(" ".join(sentences[i : i + max_chunk_size]))
        if i + max_chunk_size >= len(sentences):
            break
        i += step
    return chunks


def semantic_chunk(
    text,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int = DEFAULT_SEMANTIC_CHUNK_OVERLAP,
):
    semantic_chunks = create_semantic_chunks(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"{i}. {chunk}")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []

        for idx, doc in enumerate(documents):
            description = doc.get("description", "").strip()
            if not description:
                continue

            self.document_map[doc["id"]] = doc
            doc_chunks = create_semantic_chunks(
                description, DEFAULT_MAX_CHUNK_SIZE, DEFAULT_RAG_CHUNK_OVERLAP
            )

            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(doc_chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks),
                },
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            cached_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            cached_chunk_metadata = metadata.get("chunks", [])

            if self._chunk_cache_is_valid(
                documents,
                cached_embeddings,
                cached_chunk_metadata,
                metadata.get("total_chunks"),
            ):
                self.chunk_embeddings = cached_embeddings
                self.chunk_metadata = cached_chunk_metadata
                return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def _chunk_cache_is_valid(
        self,
        documents: list[Movie],
        chunk_embeddings: np.ndarray,
        chunk_metadata: list[ChunkMetadata],
        total_chunks: int | None,
    ) -> bool:
        if len(chunk_embeddings) != len(chunk_metadata):
            return False
        if total_chunks is not None and total_chunks != len(chunk_metadata):
            return False

        for chunk in chunk_metadata:
            movie_idx = chunk.get("movie_idx")
            chunk_idx = chunk.get("chunk_idx")
            chunk_total = chunk.get("total_chunks")

            if not isinstance(movie_idx, int) or not 0 <= movie_idx < len(documents):
                return False
            if not isinstance(chunk_idx, int) or chunk_idx < 0:
                return False
            if not isinstance(chunk_total, int) or chunk_total <= 0:
                return False
            if chunk_idx >= chunk_total:
                return False

        return True

    def search_chunks(
        self, query: str, limit: int = DEFAULT_LIB_CHUNK_SEARCH_LIMIT
    ) -> list[SearchResult]:
        if self.chunk_embeddings is None or self.chunk_embeddings.size == 0:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )
        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_chunk_embeddings` first."
            )
        if self.chunk_metadata is None or len(self.chunk_metadata) == 0:
            raise ValueError(
                "No chunk_metadata loaded. Call `load_or_create_chunk_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        chunk_scores: list[dict] = []

        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(chunk_embedding, query_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[chunk_idx]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[chunk_idx]["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores: dict[int, float] = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        ranked_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for movie_idx, score in ranked_scores[:limit]:
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc["id"],
                    doc["title"],
                    doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score,
                )
            )
        return results


def embed_chunks() -> np.ndarray:
    chunk_search_instance = ChunkedSemanticSearch()
    documents = load_movies()
    return chunk_search_instance.load_or_create_chunk_embeddings(documents)


def search_chunks_command(query: str, limit: int = DEFAULT_LIB_CHUNK_SEARCH_LIMIT):
    movies = load_movies()
    chunk_search_instance = ChunkedSemanticSearch()
    chunk_search_instance.load_or_create_chunk_embeddings(movies)
    results = chunk_search_instance.search_chunks(query, limit)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
