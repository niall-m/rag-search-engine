from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import Movie

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

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        result_pairs = self.idx.bm25_search(query, limit)
        result: list[dict] = []
        for doc_id, score in result_pairs:
            result.append({
                "id": doc_id,
                "score": score,
                "title": self.idx.docmap[doc_id]["title"],
                "description": self.idx.docmap[doc_id]["description"],
            })
        return result

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_command(nums: list[float]) -> None:
    if len(nums) == 0:
        return

    min_score = min(nums)
    max_score = max(nums)

    if min_score == max_score:
        scores = [1.0 for _ in nums]
    else:
        score_range = max_score - min_score
        scores = [(num - min_score) / score_range for num in nums]

    for score in scores:
        print(f"* {score:.4f}")
