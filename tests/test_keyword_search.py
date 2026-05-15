import unittest
import math
from io import StringIO
from unittest.mock import patch

from cli.lib.keyword_search import (
    InvertedIndex,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
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
        self.assertGreater(len(inverted_index.term_frequencies), 0)
        self.assertEqual(inverted_index.docmap[167]["title"], "Madrasapattinam")

    def test_load_raises_when_index_files_are_missing(self) -> None:
        inverted_index = InvertedIndex()

        with patch("cli.lib.keyword_search.os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                inverted_index.load()

    def test_get_tf_returns_document_term_frequency(self) -> None:
        inverted_index = InvertedIndex()

        inverted_index.load()

        self.assertEqual(inverted_index.get_tf(424, "trapper"), 4)

    def test_tf_command_prints_term_frequency(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            tf_command(424, "trapper")

        self.assertIn("Frequency of 'trapper': 4", stdout.getvalue())

    def test_get_idf_returns_inverse_document_frequency(self) -> None:
        inverted_index = InvertedIndex()

        inverted_index.load()

        term_match_doc_count = len(inverted_index.get_documents("trapper"))
        expected = math.log(
            (len(inverted_index.docmap) + 1) / (term_match_doc_count + 1)
        )
        self.assertAlmostEqual(inverted_index.get_idf("trapper"), expected)

    def test_get_idf_returns_maximum_value_for_missing_term(self) -> None:
        inverted_index = InvertedIndex()

        inverted_index.load()

        expected = math.log(len(inverted_index.docmap) + 1)
        self.assertAlmostEqual(inverted_index.get_idf("nonsenseterm"), expected)

    def test_idf_command_prints_inverse_document_frequency(self) -> None:
        inverted_index = InvertedIndex()
        inverted_index.load()
        expected_idf = inverted_index.get_idf("trapper")

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            idf_command("trapper")

        self.assertIn(
            f"Inverse document frequency of 'trapper': {expected_idf:.2f}",
            stdout.getvalue(),
        )

    def test_get_bm25_idf_returns_expected_score(self) -> None:
        inverted_index = InvertedIndex()
        inverted_index.load()

        term_match_doc_count = len(inverted_index.get_documents("trapper"))
        expected = math.log(
            (len(inverted_index.docmap) - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

        self.assertAlmostEqual(inverted_index.get_bm25_idf("trapper"), expected)

    def test_get_bm25_idf_requires_single_token(self) -> None:
        inverted_index = InvertedIndex()
        inverted_index.load()

        with self.assertRaises(ValueError):
            inverted_index.get_bm25_idf("two terms")

    def test_bm25_idf_command_prints_bm25_idf_score(self) -> None:
        inverted_index = InvertedIndex()
        inverted_index.load()
        expected_bm25_idf = inverted_index.get_bm25_idf("trapper")

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            bm25_idf_command("trapper")

        self.assertIn(
            f"BM25 IDF score of 'trapper': {expected_bm25_idf:.2f}",
            stdout.getvalue(),
        )

    def test_tfidf_command_prints_tfidf_score(self) -> None:
        inverted_index = InvertedIndex()
        inverted_index.load()
        expected_tfidf = inverted_index.get_tfidf(424, "trapper")

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            tfidf_command(424, "trapper")

        self.assertIn(
            f"TF-IDF score of 'trapper' in document '424': {expected_tfidf:.2f}",
            stdout.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
