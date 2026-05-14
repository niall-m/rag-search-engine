import unittest
from unittest.mock import patch

from cli.lib.keyword_search import (
    InvertedIndex,
    search_command,
    preprocess_text,
    tokenize_text,
)
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT


class SearchMoviesByTitleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        inverted_index = InvertedIndex()
        inverted_index.build()
        inverted_index.save()

    def test_preprocess_text_lowercases_text(self) -> None:
        self.assertEqual(preprocess_text("GrEaT"), "great")

    def test_preprocess_text_removes_punctuation(self) -> None:
        self.assertEqual(
            preprocess_text("Faster, Pussycat! Kill! Kill!"),
            "faster pussycat kill kill",
        )

    def test_tokenize_text_splits_filters_and_stems(self) -> None:
        self.assertEqual(
            tokenize_text("  the   running\tbears  "),
            ["run", "bear"],
        )

    def test_returns_first_five_results(self) -> None:
        results = search_command("brave")

        self.assertEqual(len(results), DEFAULT_SEARCH_LIMIT)
        self.assertEqual(
            [movie["title"] for movie in results],
            [
                "Madrasapattinam",
                "Imagining Argentina",
                "Cymbeline",
                "Gymkata",
                "Hornblower: Retribution",
            ],
        )

    def test_returns_first_five_results_for_second_query_token(self) -> None:
        results = search_command("nonsensetoken assault")

        self.assertEqual(len(results), DEFAULT_SEARCH_LIMIT)
        self.assertEqual(
            [movie["title"] for movie in results],
            [
                "Klansman",
                "The Chronicles of Narnia: Prince Caspian",
                "Random Hearts",
                "They Live",
                "Strange Days",
            ],
        )

    def test_search_is_case_insensitive(self) -> None:
        self.assertEqual(
            [movie["title"] for movie in search_command("BRAVE")],
            [movie["title"] for movie in search_command("brave")],
        )

    def test_load_restores_saved_index_and_docmap(self) -> None:
        inverted_index = InvertedIndex()

        inverted_index.load()

        self.assertGreater(len(inverted_index.index), 0)
        self.assertGreater(len(inverted_index.docmap), 0)
        self.assertEqual(inverted_index.docmap[167]["title"], "Madrasapattinam")

    def test_load_raises_when_index_files_are_missing(self) -> None:
        inverted_index = InvertedIndex()

        with patch("cli.lib.keyword_search.os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                inverted_index.load()


if __name__ == "__main__":
    unittest.main()
