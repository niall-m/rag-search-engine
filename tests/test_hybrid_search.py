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
    weighted_search_command,
)
from cli.lib.search_utils import (
    BM25Result,
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


if __name__ == "__main__":
    unittest.main()
