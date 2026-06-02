from typing import Literal

from .keyword_search import InvertedIndex
from .query_enhancement import rerank_results_individually
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    DEFAULT_K,
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    SEARCH_EXPANSION_MULTIPLIER,
    INDIVIDUAL_RERANK_RESULT_MULTIPLIER,
    BM25Result,
    HybridSearchResult,
    SemanticSearchResult,
    HybridRankResult,
    Movie,
    load_movies,
)


def normalize_scores(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for _ in scores]

    score_range = max_score - min_score
    return [(score - min_score) / score_range for score in scores]


def hybrid_score(
    bm25_score: float,
    semantic_score: float,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[BM25Result],
    semantic_results: list[SemanticSearchResult],
    semantic_document_map: dict[int, Movie],
    alpha: float = DEFAULT_ALPHA,
) -> list[HybridSearchResult]:
    combined_results: dict[int, HybridSearchResult] = {}

    def merge_scores(
        results: list[BM25Result] | list[SemanticSearchResult],
        normalized_scores: list[float],
        score_key: Literal["bm25_score", "semantic_score"],
    ) -> None:
        for result, normalized_score in zip(results, normalized_scores):
            description = (
                result["description"] if "description" in result else result["document"]
            )
            document = semantic_document_map.get(result["id"])
            if document is not None:
                description = document["description"]

            entry = combined_results.setdefault(
                result["id"],
                {
                    "id": result["id"],
                    "title": result["title"],
                    "description": description,
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                    "hybrid_score": 0.0,
                },
            )
            entry[score_key] = normalized_score

    merge_scores(
        bm25_results,
        normalize_scores([result["score"] for result in bm25_results]),
        "bm25_score",
    )
    merge_scores(
        semantic_results,
        normalize_scores([result["score"] for result in semantic_results]),
        "semantic_score",
    )

    for result in combined_results.values():
        result["hybrid_score"] = hybrid_score(
            result["bm25_score"],
            result["semantic_score"],
            alpha,
        )

    return sorted(
        combined_results.values(),
        key=lambda result: result["hybrid_score"],
        reverse=True,
    )


def rrf_score(rank: int, k: int = DEFAULT_K) -> float:
    return 1 / (k + rank)


def reciprocal_rank_fusion(
    bm25_results: list[BM25Result],
    semantic_results: list[SemanticSearchResult],
    semantic_document_map: dict[int, Movie],
    k: int = DEFAULT_K,
) -> list[HybridRankResult]:
    combined_results: dict[int, HybridRankResult] = {}

    def merge_ranks(
        results: list[BM25Result] | list[SemanticSearchResult],
        rank_key: Literal["bm25_rank", "semantic_rank"],
    ) -> None:
        for rank, result in enumerate(results, 1):
            description = (
                result["description"] if "description" in result else result["document"]
            )
            document = semantic_document_map.get(result["id"])
            if document is not None:
                description = document["description"]

            entry = combined_results.setdefault(
                result["id"],
                {
                    "id": result["id"],
                    "title": result["title"],
                    "description": description,
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                },
            )
            entry[rank_key] = rank
            entry["rrf_score"] += rrf_score(rank, k)

    merge_ranks(
        bm25_results,
        "bm25_rank",
    )
    merge_ranks(
        semantic_results,
        "semantic_rank",
    )

    return sorted(
        combined_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        try:
            self.idx.load()
        except FileNotFoundError:
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[BM25Result]:
        self.idx.load()
        result_pairs = self.idx.bm25_search(query, limit)
        result: list[BM25Result] = []
        for doc_id, score in result_pairs:
            result.append(
                {
                    "id": doc_id,
                    "score": score,
                    "title": self.idx.docmap[doc_id]["title"],
                    "description": self.idx.docmap[doc_id]["description"],
                }
            )
        return result

    def weighted_search(
        self, query: str, alpha: float = DEFAULT_ALPHA, limit: int = 5
    ) -> list[HybridSearchResult]:
        expanded_limit = limit * SEARCH_EXPANSION_MULTIPLIER
        bm25_results = self._bm25_search(query, expanded_limit)
        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)
        return combine_search_results(
            bm25_results,
            semantic_results,
            self.semantic_search.document_map,
            alpha,
        )

    def rrf_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        limit: int = DEFAULT_SEARCH_LIMIT,
        rerank_method: Literal["individual"] | None = None,
    ) -> list[HybridRankResult]:
        expanded_limit = limit * SEARCH_EXPANSION_MULTIPLIER
        bm25_results = self._bm25_search(query, expanded_limit)
        semantic_results = self.semantic_search.search_chunks(query, expanded_limit)
        fused = reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            self.semantic_search.document_map,
            k,
        )

        if rerank_method == "individual":
            result_limit = limit * INDIVIDUAL_RERANK_RESULT_MULTIPLIER
            fused = fused[:result_limit]
            fused = rerank_results_individually(query, fused)

        return fused[:limit]


def normalize_command(nums: list[float]) -> None:
    for score in normalize_scores(nums):
        print(f"* {score:.4f}")


def weighted_search_command(
    query: str,
    alpha: float = DEFAULT_ALPHA,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[HybridSearchResult]:
    search = HybridSearch(load_movies())
    return search.weighted_search(query, alpha, limit)


def rrf_search_command(
    query: str,
    k: int = DEFAULT_K,
    limit: int = DEFAULT_SEARCH_LIMIT,
    rerank_method: Literal["individual"] | None = None,
) -> list[HybridRankResult]:
    search = HybridSearch(load_movies())
    return search.rrf_search(query, k, limit, rerank_method)
