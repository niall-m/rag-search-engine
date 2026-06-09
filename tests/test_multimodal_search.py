import importlib
import sys
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from cli.lib import multimodal_search

CLI_DIR = Path(__file__).resolve().parents[1] / "cli"
sys.path.insert(0, str(CLI_DIR))

multimodal_search_cli = importlib.import_module("multimodal_search_cli")

TEST_DOCUMENTS = [
    {
        "id": 1,
        "title": "Arrival",
        "description": "Linguists meet aliens.",
    },
    {
        "id": 2,
        "title": "Paddington",
        "description": "A bear travels to London.",
    },
]


class MultimodalSearchTests(unittest.TestCase):
    def test_constructor_loads_default_documents_builds_texts_and_embeddings(
        self,
    ) -> None:
        fake_embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

        with (
            patch.object(
                multimodal_search,
                "load_movies",
                return_value=TEST_DOCUMENTS,
            ),
            patch.object(multimodal_search, "SentenceTransformer") as model_class,
        ):
            model_class.return_value.encode.return_value = fake_embeddings

            search = multimodal_search.MultimodalSearch()

        model_class.assert_called_once_with("clip-ViT-B-32")
        model_class.return_value.encode.assert_called_once_with(
            [
                "Arrival: Linguists meet aliens.",
                "Paddington: A bear travels to London.",
            ],
            show_progress_bar=True,
        )
        self.assertEqual(search.documents, TEST_DOCUMENTS)
        self.assertEqual(
            search.texts,
            [
                "Arrival: Linguists meet aliens.",
                "Paddington: A bear travels to London.",
            ],
        )
        self.assertTrue(np.array_equal(search.text_embeddings, fake_embeddings))

    def test_search_with_image_scores_all_documents_sorts_and_returns_top_five(
        self,
    ) -> None:
        search = object.__new__(multimodal_search.MultimodalSearch)
        search.documents = [
            {"id": 1, "title": "One", "description": "First"},
            {"id": 2, "title": "Two", "description": "Second"},
            {"id": 3, "title": "Three", "description": "Third"},
            {"id": 4, "title": "Four", "description": "Fourth"},
            {"id": 5, "title": "Five", "description": "Fifth"},
            {"id": 6, "title": "Six", "description": "Sixth"},
        ]
        search.text_embeddings = np.array(
            [
                [0.1, 0.9],
                [0.9, 0.1],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.7, 0.7],
                [0.8, 0.2],
            ]
        )
        search.embed_image = MagicMock(return_value=np.array([1.0, 0.0]))

        results = search.search_with_image("poster.jpg")

        search.embed_image.assert_called_once_with("poster.jpg")
        self.assertEqual([result["id"] for result in results], [3, 2, 6, 5, 1])
        self.assertEqual(len(results), 5)
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0)
        self.assertLess(results[-1]["similarity_score"], results[0]["similarity_score"])

    def test_embed_image_raises_for_missing_file(self) -> None:
        search = object.__new__(multimodal_search.MultimodalSearch)

        with self.assertRaisesRegex(FileNotFoundError, "Image file not found"):
            search.embed_image("missing.jpg")

    def test_image_search_command_creates_search_and_returns_results(self) -> None:
        expected_results = [
            {
                "id": 2,
                "title": "Paddington",
                "description": "A bear travels to London.",
                "similarity_score": 0.722,
            }
        ]

        with patch.object(multimodal_search, "MultimodalSearch") as search_class:
            search_class.return_value.search_with_image.return_value = expected_results

            results = multimodal_search.image_search_command("poster.jpg")

        search_class.assert_called_once_with()
        search_class.return_value.search_with_image.assert_called_once_with(
            "poster.jpg"
        )
        self.assertEqual(results, expected_results)


class MultimodalSearchCliTests(unittest.TestCase):
    def test_run_command_prints_ranked_image_search_results(self) -> None:
        args = SimpleNamespace(command="image_search", image_path="poster.jpg")
        search_results = [
            {
                "id": 2,
                "title": "Paddington",
                "description": "Deep in the rainforests of Peru, a young bear lives peacefully.",
                "similarity_score": 0.7224,
            }
        ]

        with (
            patch.object(
                multimodal_search_cli,
                "image_search_command",
                return_value=search_results,
            ),
            patch.object(multimodal_search_cli, "DOCUMENT_PREVIEW_LENGTH", 20),
            patch("sys.stdout", new_callable=StringIO) as stdout,
        ):
            multimodal_search_cli.run_command(args)

        self.assertEqual(
            stdout.getvalue(),
            "1. Paddington (similarity: 0.722)\n"
            "   Deep in the rainfore...\n\n",
        )


if __name__ == "__main__":
    unittest.main()
