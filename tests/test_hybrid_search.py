import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from cli.lib.hybrid_search import (
    HybridSearch,
    combine_search_results,
    hybrid_score,
    normalize_command,
    normalize_scores,
    reciprocal_rank_fusion,
    rrf_score,
    rrf_search_command,
    weighted_search_command,
)
from cli.lib.query_enhancement import rerank_batch
from cli.lib.search_utils import (
    BM25Result,
    HybridRankResult,
    HybridSearchResult,
    Movie,
    SemanticSearchResult,
)


class NormalizeCommandTests(unittest.TestCase):
    def test_normalize_scores_returns_normalized_values(self) -> None:
        actual_scores = normalize_scores([0.5, 2.3, 1.2, 0.5, 0.1])
        expected_scores = [
            0.18181818181818182,
            1.0,
            0.5,
            0.18181818181818182,
            0.0,
        ]

        for actual_score, expected_score in zip(actual_scores, expected_scores):
            self.assertAlmostEqual(actual_score, expected_score)

    def test_normalize_command_prints_min_max_normalized_scores_for_ints(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            normalize_command([1, 2, 3])

        self.assertEqual(
            stdout.getvalue(),
            "* 0.0000\n* 0.5000\n* 1.0000\n",
        )

    def test_normalize_command_prints_ones_when_all_scores_match(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            normalize_command([1, 1, 1])

        self.assertEqual(
            stdout.getvalue(),
            "* 1.0000\n* 1.0000\n* 1.0000\n",
        )

    def test_normalize_command_prints_nothing_for_empty_input(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            normalize_command([])

        self.assertEqual(stdout.getvalue(), "")

    def test_hybrid_score_uses_alpha_weighting(self) -> None:
        self.assertAlmostEqual(hybrid_score(0.8, 0.2, alpha=0.75), 0.65)


class WeightedSearchTests(unittest.TestCase):
    def semantic_document_map(self) -> dict[int, Movie]:
        return {
            1: {
                "id": 1,
                "title": "Alpha",
                "description": "Alpha description",
            },
            2: {
                "id": 2,
                "title": "Bravo",
                "description": "Bravo description",
            },
            3: {
                "id": 3,
                "title": "Charlie",
                "description": "Charlie description",
            },
        }

    def create_search_instance(self) -> HybridSearch:
        search = object.__new__(HybridSearch)
        search.semantic_search = SimpleNamespace(
            document_map=self.semantic_document_map()
        )
        return search

    def sample_bm25_results(self) -> list[BM25Result]:
        return [
            {
                "id": 1,
                "score": 30.0,
                "title": "Alpha",
                "description": "Alpha description",
            },
            {
                "id": 3,
                "score": 20.0,
                "title": "Charlie",
                "description": "Charlie description",
            },
            {
                "id": 2,
                "score": 10.0,
                "title": "Bravo",
                "description": "Bravo description",
            },
        ]

    def sample_semantic_results(self) -> list[SemanticSearchResult]:
        return [
            {
                "id": 2,
                "title": "Bravo",
                "document": "Bravo description",
                "score": 0.9,
                "metadata": {},
            },
            {
                "id": 3,
                "title": "Charlie",
                "document": "Charlie description",
                "score": 0.5,
                "metadata": {},
            },
            {
                "id": 1,
                "title": "Alpha",
                "document": "Alpha description",
                "score": 0.2,
                "metadata": {},
            },
        ]

    def test_combine_search_results_normalizes_combines_and_sorts_results(
        self,
    ) -> None:
        results = combine_search_results(
            self.sample_bm25_results(),
            self.sample_semantic_results(),
            self.semantic_document_map(),
            alpha=0.25,
        )

        self.assertEqual([result["id"] for result in results], [2, 3, 1])
        self.assertAlmostEqual(results[0]["bm25_score"], 0.0)
        self.assertAlmostEqual(results[0]["semantic_score"], 1.0)
        self.assertAlmostEqual(results[0]["hybrid_score"], 0.75)
        self.assertEqual(results[0]["description"], "Bravo description")

    def test_weighted_search_normalizes_combines_and_sorts_results(self) -> None:
        search = self.create_search_instance()
        bm25_results = self.sample_bm25_results()
        semantic_results = self.sample_semantic_results()

        with (
            patch.object(
                search,
                "_bm25_search",
                return_value=bm25_results,
            ) as bm25_search,
            patch.object(
                search.semantic_search,
                "search_chunks",
                return_value=semantic_results,
                create=True,
            ) as semantic_search,
        ):
            results = search.weighted_search("family movies", alpha=0.25, limit=2)

        bm25_search.assert_called_once_with("family movies", 1000)
        semantic_search.assert_called_once_with("family movies", 1000)
        self.assertEqual([result["id"] for result in results], [2, 3, 1])
        self.assertAlmostEqual(results[0]["bm25_score"], 0.0)
        self.assertAlmostEqual(results[0]["semantic_score"], 1.0)
        self.assertAlmostEqual(results[0]["hybrid_score"], 0.75)
        self.assertEqual(results[0]["description"], "Bravo description")

    def test_weighted_search_command_loads_movies_and_runs_search(self) -> None:
        weighted_results: list[HybridSearchResult] = [
            {
                "id": 2,
                "title": "Bravo",
                "description": "Bravo description",
                "bm25_score": 0.0,
                "semantic_score": 1.0,
                "hybrid_score": 0.75,
            }
        ]

        with (
            patch("cli.lib.hybrid_search.load_movies", return_value=[]),
            patch("cli.lib.hybrid_search.HybridSearch") as hybrid_search_class,
        ):
            hybrid_search_class.return_value.weighted_search.return_value = (
                weighted_results
            )

            results = weighted_search_command("family movies", alpha=0.25, limit=1)

        hybrid_search_class.assert_called_once_with([])
        hybrid_search_class.return_value.weighted_search.assert_called_once_with(
            "family movies",
            0.25,
            1,
        )
        self.assertEqual(results, weighted_results)


class ReciprocalRankFusionTests(unittest.TestCase):
    def semantic_document_map(self) -> dict[int, Movie]:
        return {
            1: {
                "id": 1,
                "title": "Alpha",
                "description": "Alpha description",
            },
            2: {
                "id": 2,
                "title": "Bravo",
                "description": "Bravo description",
            },
            3: {
                "id": 3,
                "title": "Charlie",
                "description": "Charlie description",
            },
        }

    def create_search_instance(self) -> HybridSearch:
        search = object.__new__(HybridSearch)
        search.semantic_search = SimpleNamespace(
            document_map=self.semantic_document_map()
        )
        return search

    def test_rrf_score_uses_k_and_rank(self) -> None:
        self.assertAlmostEqual(rrf_score(1, k=20), 1 / 21)
        self.assertAlmostEqual(rrf_score(5, k=60), 1 / 65)

    def test_reciprocal_rank_fusion_combines_rank_positions_and_sorts(self) -> None:
        bm25_results: list[BM25Result] = [
            {
                "id": 1,
                "score": 30.0,
                "title": "Alpha",
                "description": "Alpha description",
            },
            {
                "id": 2,
                "score": 20.0,
                "title": "Bravo",
                "description": "Bravo description",
            },
            {
                "id": 3,
                "score": 10.0,
                "title": "Charlie",
                "description": "Charlie description",
            },
        ]
        semantic_results: list[SemanticSearchResult] = [
            {
                "id": 1,
                "title": "Alpha",
                "document": "Alpha description",
                "score": 0.9,
                "metadata": {},
            },
            {
                "id": 3,
                "title": "Charlie",
                "document": "Charlie description",
                "score": 0.8,
                "metadata": {},
            },
        ]

        results = reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            self.semantic_document_map(),
            k=60,
        )

        self.assertEqual([result["id"] for result in results], [1, 3, 2])
        self.assertEqual(results[0]["bm25_rank"], 1)
        self.assertEqual(results[0]["semantic_rank"], 1)
        self.assertAlmostEqual(results[0]["rrf_score"], 2 / 61)
        self.assertEqual(results[1]["bm25_rank"], 3)
        self.assertEqual(results[1]["semantic_rank"], 2)
        self.assertAlmostEqual(results[1]["rrf_score"], (1 / 63) + (1 / 62))
        self.assertEqual(results[2]["bm25_rank"], 2)
        self.assertIsNone(results[2]["semantic_rank"])
        self.assertAlmostEqual(results[2]["rrf_score"], 1 / 62)

    def test_rrf_search_expands_limit_and_uses_chunked_semantic_search(self) -> None:
        search = self.create_search_instance()
        bm25_results: list[BM25Result] = [
            {
                "id": 1,
                "score": 10.0,
                "title": "Alpha",
                "description": "Alpha description",
            }
        ]
        semantic_results: list[SemanticSearchResult] = [
            {
                "id": 1,
                "title": "Alpha",
                "document": "Alpha description",
                "score": 0.9,
                "metadata": {},
            }
        ]

        with (
            patch.object(
                search, "_bm25_search", return_value=bm25_results
            ) as bm25_search,
            patch.object(
                search.semantic_search,
                "search_chunks",
                return_value=semantic_results,
                create=True,
            ) as semantic_search,
        ):
            results = search.rrf_search("family movies", k=30, limit=2)

        bm25_search.assert_called_once_with("family movies", 1000)
        semantic_search.assert_called_once_with("family movies", 1000)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["bm25_rank"], 1)
        self.assertEqual(results[0]["semantic_rank"], 1)
        self.assertAlmostEqual(results[0]["rrf_score"], 2 / 31)

    def test_rrf_search_only_calls_rerank_when_method_is_provided(self) -> None:
        search = self.create_search_instance()
        bm25_results: list[BM25Result] = [
            {
                "id": 1,
                "score": 10.0,
                "title": "Alpha",
                "description": "Alpha description",
            }
        ]
        semantic_results: list[SemanticSearchResult] = [
            {
                "id": 1,
                "title": "Alpha",
                "document": "Alpha description",
                "score": 0.9,
                "metadata": {},
            }
        ]

        with (
            patch.object(search, "_bm25_search", return_value=bm25_results),
            patch.object(
                search.semantic_search,
                "search_chunks",
                return_value=semantic_results,
                create=True,
            ),
            patch("cli.lib.hybrid_search.rerank") as rerank_fn,
        ):
            results = search.rrf_search("family movies", k=30, limit=2)

        rerank_fn.assert_not_called()
        self.assertEqual(results[0]["id"], 1)

    def test_rrf_search_command_loads_movies_and_runs_search(self) -> None:
        rrf_results: list[HybridRankResult] = [
            {
                "id": 2,
                "title": "Bravo",
                "description": "Bravo description",
                "bm25_rank": 2,
                "semantic_rank": 1,
                "rrf_score": 0.0325,
            }
        ]

        with (
            patch("cli.lib.hybrid_search.load_movies", return_value=[]),
            patch("cli.lib.hybrid_search.HybridSearch") as hybrid_search_class,
        ):
            hybrid_search_class.return_value.rrf_search.return_value = rrf_results

            results = rrf_search_command("family movies", k=30, limit=1)

        hybrid_search_class.assert_called_once_with([])
        hybrid_search_class.return_value.rrf_search.assert_called_once_with(
            "family movies",
            30,
            1,
        )
        self.assertEqual(results, rrf_results)


class BatchRerankTests(unittest.TestCase):
    def test_rerank_batch_assigns_rank_and_sorts_results(self) -> None:
        results: list[HybridRankResult] = [
            {
                "id": 1,
                "title": "Alpha",
                "description": "Alpha description",
                "bm25_rank": 1,
                "semantic_rank": 3,
                "rrf_score": 0.03,
            },
            {
                "id": 2,
                "title": "Bravo",
                "description": "Bravo description",
                "bm25_rank": 2,
                "semantic_rank": 2,
                "rrf_score": 0.02,
            },
            {
                "id": 3,
                "title": "Charlie",
                "description": "Charlie description",
                "bm25_rank": 3,
                "semantic_rank": 1,
                "rrf_score": 0.01,
            },
        ]
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_: SimpleNamespace(text="[3, 1, 2]")
            )
        )

        with patch("cli.lib.query_enhancement._create_client", return_value=client):
            reranked_results = rerank_batch("family movies", results)

        self.assertEqual([result["id"] for result in reranked_results], [3, 1, 2])
        self.assertEqual(
            [result["rerank_rank"] for result in reranked_results],
            [1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
