import importlib
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

CLI_DIR = Path(__file__).resolve().parents[1] / "cli"
sys.path.insert(0, str(CLI_DIR))

evaluation_cli = importlib.import_module("evaluation_cli")

calculate_f1_score = evaluation_cli.calculate_f1_score
calculate_precision_at_k = evaluation_cli.calculate_precision_at_k
calculate_recall_at_k = evaluation_cli.calculate_recall_at_k
evaluate_search_at_k = evaluation_cli.evaluate_search_at_k
print_evaluation_results = evaluation_cli.print_evaluation_results
DEFAULT_K = evaluation_cli.DEFAULT_K


class FakeSearch:
    def __init__(self, results_by_query: dict[str, list[dict[str, object]]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int, int]] = []

    def rrf_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        self.calls.append((query, k, limit))
        return self.results_by_query[query]


class EvaluationCliTests(unittest.TestCase):
    def test_calculate_precision_at_k_counts_relevant_retrieved_titles(self) -> None:
        precision = calculate_precision_at_k(
            ["The Edge", "Alaska", "Paddington"],
            ["Alaska", "The Revenant"],
        )

        self.assertAlmostEqual(precision, 1 / 3)

    def test_calculate_recall_at_k_counts_relevant_titles_found_in_results(
        self,
    ) -> None:
        recall = calculate_recall_at_k(
            ["The Edge", "Alaska", "Paddington"],
            ["The Edge", "Alaska", "The Revenant", "Grizzly Man"],
        )

        self.assertAlmostEqual(recall, 0.5)

    def test_calculate_f1_score_balances_precision_and_recall(self) -> None:
        self.assertAlmostEqual(calculate_f1_score(0.75, 0.5), 0.6)
        self.assertEqual(calculate_f1_score(0.0, 0.0), 0.0)

    def test_evaluate_search_at_k_uses_rrf_search_and_preserves_relevant_titles(
        self,
    ) -> None:
        test_cases = [
            {
                "query": "dangerous bear wilderness survival",
                "relevant_docs": ["The Edge", "Alaska", "The Revenant"],
            },
            {
                "query": "cute british bear marmalade",
                "relevant_docs": ["Paddington"],
            },
        ]
        search = FakeSearch(
            {
                "dangerous bear wilderness survival": [
                    {"title": "The Edge"},
                    {"title": "Alaska"},
                ],
                "cute british bear marmalade": [
                    {"title": "Paddington"},
                    {"title": "The Bear"},
                ],
            }
        )

        results = evaluate_search_at_k(search, test_cases, limit=2)

        self.assertEqual(
            search.calls,
            [
                ("dangerous bear wilderness survival", DEFAULT_K, 2),
                ("cute british bear marmalade", DEFAULT_K, 2),
            ],
        )
        self.assertEqual(results[0]["retrieved_titles"], ["The Edge", "Alaska"])
        self.assertEqual(
            results[0]["relevant_titles"],
            ["The Edge", "Alaska", "The Revenant"],
        )
        self.assertAlmostEqual(results[0]["precision"], 1.0)
        self.assertAlmostEqual(results[0]["recall"], 2 / 3)
        self.assertAlmostEqual(results[0]["f1"], 0.8)
        self.assertAlmostEqual(results[1]["precision"], 0.5)
        self.assertAlmostEqual(results[1]["recall"], 1.0)
        self.assertAlmostEqual(results[1]["f1"], 2 / 3)

    def test_print_evaluation_results_matches_expected_output(self) -> None:
        results = [
            {
                "query": "dangerous bear wilderness survival",
                "precision": 1.0,
                "recall": 6 / 7,
                "f1": 12 / 13,
                "retrieved_titles": [
                    "The Edge",
                    "Man in the Wilderness",
                    "Claws",
                    "Unnatural",
                    "Into the Grizzly Maze",
                    "Alaska",
                ],
                "relevant_titles": [
                    "Unnatural",
                    "Alaska",
                    "The Edge",
                    "Into the Grizzly Maze",
                    "Claws",
                    "Man in the Wilderness",
                    "The Revenant",
                ],
            },
            {
                "query": "cute british bear marmalade",
                "precision": 1 / 6,
                "recall": 1.0,
                "f1": 2 / 7,
                "retrieved_titles": [
                    "Paddington",
                    "The Indian in the Cupboard",
                    "The Duchess",
                    "The Great Bear",
                    "The Bear",
                    "Goldilocks and the Three Bears",
                ],
                "relevant_titles": ["Paddington"],
            },
        ]

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            print_evaluation_results(results, limit=6)

        self.assertEqual(
            stdout.getvalue(),
            "k=6\n"
            "\n"
            "- Query: dangerous bear wilderness survival\n"
            "  - Precision@6: 1.0000\n"
            "  - Recall@6: 0.8571\n"
            "  - F1 Score: 0.9231\n"
            "  - Retrieved: The Edge, Man in the Wilderness, Claws, Unnatural, "
            "Into the Grizzly Maze, Alaska\n"
            "  - Relevant: Unnatural, Alaska, The Edge, Into the Grizzly Maze, "
            "Claws, Man in the Wilderness, The Revenant\n"
            "\n"
            "- Query: cute british bear marmalade\n"
            "  - Precision@6: 0.1667\n"
            "  - Recall@6: 1.0000\n"
            "  - F1 Score: 0.2857\n"
            "  - Retrieved: Paddington, The Indian in the Cupboard, The Duchess, "
            "The Great Bear, The Bear, Goldilocks and the Three Bears\n"
            "  - Relevant: Paddington\n"
            "\n",
        )
