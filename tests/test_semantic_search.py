import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from cli.lib import semantic_search


TEST_DOCUMENTS = [
    {"id": 1, "title": "Arrival", "description": "Linguists meet aliens."},
    {"id": 2, "title": "Primer", "description": "Engineers invent time travel."},
]


class FakeModel:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
        self.encode_calls: list[tuple[list[str], bool]] = []

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        self.encode_calls.append((texts, show_progress_bar))
        return self.embeddings


class SemanticSearchTests(unittest.TestCase):
    def create_search_instance(self, model: FakeModel | None = None):
        search = object.__new__(semantic_search.SemanticSearch)
        search.model = model
        search.embeddings = None
        search.documents = None
        search.document_map = {999: {"id": 999, "title": "stale", "description": "old"}}
        return search

    def test_build_embeddings_saves_embeddings_and_resets_document_map(self) -> None:
        expected_embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = FakeModel(expected_embeddings)
        search = self.create_search_instance(model)

        with TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            embeddings_path = cache_dir / "movie_embeddings.npy"

            with (
                patch.object(semantic_search, "CACHE_DIR", cache_dir),
                patch.object(semantic_search, "MOVIE_EMBEDDINGS_PATH", embeddings_path),
            ):
                embeddings = search.build_embeddings(TEST_DOCUMENTS)
                self.assertTrue(embeddings_path.exists())

        self.assertTrue(np.array_equal(embeddings, expected_embeddings))
        self.assertEqual(search.documents, TEST_DOCUMENTS)
        self.assertEqual(
            search.document_map, {1: TEST_DOCUMENTS[0], 2: TEST_DOCUMENTS[1]}
        )
        self.assertEqual(
            model.encode_calls,
            [
                (
                    [
                        "Arrival: Linguists meet aliens.",
                        "Primer: Engineers invent time travel.",
                    ],
                    True,
                )
            ],
        )

    def test_load_or_create_embeddings_uses_cached_embeddings_without_rebuilding(
        self,
    ) -> None:
        cached_embeddings = np.array([[10.0, 20.0], [30.0, 40.0]])
        search = self.create_search_instance()

        with TemporaryDirectory() as temp_dir:
            embeddings_path = Path(temp_dir) / "movie_embeddings.npy"
            np.save(embeddings_path, cached_embeddings)

            with (
                patch.object(semantic_search, "MOVIE_EMBEDDINGS_PATH", embeddings_path),
                patch.object(
                    search,
                    "build_embeddings",
                    side_effect=AssertionError("build_embeddings should not run"),
                ),
            ):
                embeddings = search.load_or_create_embeddings(TEST_DOCUMENTS)

        self.assertTrue(np.array_equal(embeddings, cached_embeddings))
        self.assertEqual(search.embeddings.shape, (2, 2))
        self.assertEqual(search.documents, TEST_DOCUMENTS)
        self.assertEqual(
            search.document_map, {1: TEST_DOCUMENTS[0], 2: TEST_DOCUMENTS[1]}
        )

    def test_verify_embeddings_prints_document_and_embedding_sizes(self) -> None:
        fake_embeddings = np.zeros((2, 3))

        with (
            patch.object(semantic_search, "load_movies", return_value=TEST_DOCUMENTS),
            patch.object(semantic_search, "SemanticSearch") as semantic_search_class,
            patch("sys.stdout", new_callable=StringIO) as stdout,
        ):
            semantic_search_class.return_value.load_or_create_embeddings.return_value = fake_embeddings

            semantic_search.verify_embeddings()

        output = stdout.getvalue()
        self.assertIn("Number of docs:   2", output)
        self.assertIn("Embeddings shape: 2 vectors in 3 dimensions", output)

    def test_search_raises_if_embeddings_are_not_loaded(self) -> None:
        search = self.create_search_instance()

        with self.assertRaisesRegex(
            ValueError,
            "No embeddings loaded. Call `load_or_create_embeddings` first.",
        ):
            search.search("aliens", limit=2)

    def test_search_returns_ranked_results_with_scores(self) -> None:
        search = self.create_search_instance()
        search.documents = TEST_DOCUMENTS
        search.document_map = {1: TEST_DOCUMENTS[0], 2: TEST_DOCUMENTS[1]}
        search.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        search.generate_embedding = lambda query: np.array([1.0, 0.0])

        results = search.search("first movie", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Arrival")
        self.assertEqual(results[0]["description"], "Linguists meet aliens.")
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_create_chunks_supports_overlap_and_short_inputs(self) -> None:
        overlapping_chunks = semantic_search.create_chunks(
            "This is a test text with two chunks",
            chunk_size=5,
            overlap=2,
        )

        self.assertEqual(
            overlapping_chunks,
            [
                "This is a test text",
                "test text with two chunks",
            ],
        )

        short_chunks = semantic_search.create_chunks("only three words")

        self.assertEqual(short_chunks, ["only three words"])

    def test_semantic_chunk_prints_expected_chunks(self) -> None:
        text = (
            "This is the first sentence. This is the second sentence. "
            "This is the third sentence. This is the fourth sentence. "
            "This is the fifth sentence."
        )

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            semantic_search.semantic_chunk(text, max_chunk_size=3, overlap=0)

        output = stdout.getvalue()
        self.assertIn("Semantically chunking 141 characters", output)
        self.assertIn(
            "1. This is the first sentence. This is the second sentence. "
            "This is the third sentence.",
            output,
        )
        self.assertIn(
            "2. This is the fourth sentence. This is the fifth sentence.",
            output,
        )


if __name__ == "__main__":
    unittest.main()
