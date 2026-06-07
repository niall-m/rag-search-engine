import importlib
import sys
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli.lib.augmented_generation import MODEL, rag_search_command
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT

CLI_DIR = Path(__file__).resolve().parents[1] / "cli"
sys.path.insert(0, str(CLI_DIR))

augmented_generation_cli = importlib.import_module("augmented_generation_cli")


class AugmentedGenerationTests(unittest.TestCase):
    def test_rag_search_command_formats_documents_for_prompt(self) -> None:
        search_results = [
            {
                "id": 1,
                "title": "Jurassic Park",
                "description": "Dinosaurs escape a theme park.",
                "bm25_rank": 1,
                "semantic_rank": 2,
                "rrf_score": 0.0325,
            },
            {
                "id": 2,
                "title": "The Lost World",
                "description": "A team returns to a dinosaur island.",
                "bm25_rank": 2,
                "semantic_rank": 1,
                "rrf_score": 0.0325,
            },
        ]
        client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_: SimpleNamespace(text="Generated answer")
            )
        )

        with (
            patch("cli.lib.augmented_generation.load_movies", return_value=[]),
            patch("cli.lib.augmented_generation.HybridSearch") as hybrid_search_class,
            patch("cli.lib.augmented_generation.create_client", return_value=client),
        ):
            hybrid_search_class.return_value.rrf_search.return_value = search_results

            result = rag_search_command("dinosaur movies")

        hybrid_search_class.assert_called_once_with([])
        hybrid_search_class.return_value.rrf_search.assert_called_once_with(
            "dinosaur movies",
            limit=DEFAULT_SEARCH_LIMIT,
        )
        self.assertEqual(result["search_results"], search_results)
        self.assertEqual(result["llm_response"], "Generated answer")

    def test_rag_search_command_uses_clean_document_context_in_prompt(self) -> None:
        search_results = [
            {
                "id": 1,
                "title": "Jurassic Park",
                "description": "Dinosaurs escape a theme park.",
                "bm25_rank": 1,
                "semantic_rank": 2,
                "rrf_score": 0.0325,
            }
        ]
        response = SimpleNamespace(text="Generated answer")
        generate_content = MagicMock(return_value=response)
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

        with (
            patch("cli.lib.augmented_generation.load_movies", return_value=[]),
            patch("cli.lib.augmented_generation.HybridSearch") as hybrid_search_class,
            patch("cli.lib.augmented_generation.create_client", return_value=client),
        ):
            hybrid_search_class.return_value.rrf_search.return_value = search_results
            rag_search_command("dinosaur movies")

        generate_content.assert_called_once()
        prompt = generate_content.call_args.kwargs["contents"]
        self.assertEqual(generate_content.call_args.kwargs["model"], MODEL)
        self.assertIn("Query: dinosaur movies", prompt)
        self.assertIn("1. Jurassic Park", prompt)
        self.assertIn("Description: Dinosaurs escape a theme park.", prompt)
        self.assertIn("Use only the retrieved documents below.", prompt)
        self.assertIn(
            "If the documents are insufficient to fully answer the query, say so.",
            prompt,
        )
        self.assertNotIn("'rrf_score'", prompt)
        self.assertNotIn("'bm25_rank'", prompt)


class AugmentedGenerationCliTests(unittest.TestCase):
    def test_run_command_prints_search_results_and_rag_response(self) -> None:
        args = SimpleNamespace(command="rag", query="dinosaur movies")
        result = {
            "search_results": [
                {"title": "We're Back! A Dinosaur's Story"},
                {"title": "Jurassic Park"},
                {"title": "The Lost World"},
                {"title": "Carnosaur"},
                {"title": "A Sound of Thunder"},
            ],
            "llm_response": "Here are some dinosaur movies you might enjoy.",
        }

        with (
            patch.object(
                augmented_generation_cli,
                "rag_search_command",
                return_value=result,
            ),
            patch("sys.stdout", new_callable=StringIO) as stdout,
        ):
            augmented_generation_cli.run_command(args)

        self.assertEqual(
            stdout.getvalue(),
            "Search Results:\n"
            "- We're Back! A Dinosaur's Story\n"
            "- Jurassic Park\n"
            "- The Lost World\n"
            "- Carnosaur\n"
            "- A Sound of Thunder\n"
            "\n"
            "RAG Response:\n"
            "Here are some dinosaur movies you might enjoy.\n",
        )
