import unittest
import math
import subprocess
import sys
from io import StringIO
from unittest.mock import patch

from cli.lib.keyword_search import (
    InvertedIndex,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    search_command,
    preprocess_text,
    tokenize_text,
)
from cli.lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    PROJECT_ROOT,
    MOVIES_DATA_PATH,
    STOPWORDS_DATA_PATH,
    INDEX_DISK_DATA_PATH,
    DOCMAP_DISK_DATA_PATH,
    TF_DISK_DATA_PATH,
    DOC_LENGTHS_PATH,
)


CACHE_PATHS = (
    INDEX_DISK_DATA_PATH,
    DOCMAP_DISK_DATA_PATH,
    TF_DISK_DATA_PATH,
    DOC_LENGTHS_PATH,
)
INDEX_SOURCE_PATHS = (
    MOVIES_DATA_PATH,
    STOPWORDS_DATA_PATH,
    PROJECT_ROOT / "cli" / "lib" / "keyword_search.py",
)


def cache_is_fresh() -> bool:
    if any(not path.exists() for path in CACHE_PATHS):
        return False

    newest_source_mtime = max(path.stat().st_mtime for path in INDEX_SOURCE_PATHS)
    oldest_cache_mtime = min(path.stat().st_mtime for path in CACHE_PATHS)
    return oldest_cache_mtime >= newest_source_mtime


def load_or_build_test_index() -> InvertedIndex:
    inverted_index = InvertedIndex()

    if cache_is_fresh():
        inverted_index.load()
        return inverted_index

    inverted_index.build()
    inverted_index.save()
    return inverted_index


class SearchMoviesByTitleTests(unittest.TestCase):
    inverted_index: InvertedIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.inverted_index = load_or_build_test_index()

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
        self.assertEqual(self.inverted_index.get_tf(424, "trapper"), 4)

    def test_tf_command_prints_term_frequency(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            tf_command(424, "trapper")

        self.assertIn("Frequency of 'trapper': 4", stdout.getvalue())

    def test_get_avg_doc_length_returns_average_length(self) -> None:
        expected = sum(self.inverted_index.doc_lengths.values()) / len(
            self.inverted_index.doc_lengths
        )
        get_avg_doc_length = getattr(
            self.inverted_index, "_InvertedIndex__get_avg_doc_length"
        )

        self.assertAlmostEqual(
            get_avg_doc_length(),
            expected,
        )

    def test_get_avg_doc_length_returns_zero_for_empty_index(self) -> None:
        inverted_index = InvertedIndex()
        get_avg_doc_length = getattr(
            inverted_index, "_InvertedIndex__get_avg_doc_length"
        )

        self.assertEqual(get_avg_doc_length(), 0.0)

    def test_get_idf_returns_inverse_document_frequency(self) -> None:
        term_match_doc_count = len(self.inverted_index.get_documents("trapper"))
        expected = math.log(
            (len(self.inverted_index.docmap) + 1) / (term_match_doc_count + 1)
        )
        self.assertAlmostEqual(self.inverted_index.get_idf("trapper"), expected)

    def test_get_idf_returns_maximum_value_for_missing_term(self) -> None:
        expected = math.log(len(self.inverted_index.docmap) + 1)
        self.assertAlmostEqual(self.inverted_index.get_idf("nonsenseterm"), expected)

    def test_idf_command_prints_inverse_document_frequency(self) -> None:
        expected_idf = self.inverted_index.get_idf("trapper")

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            idf_command("trapper")

        self.assertIn(
            f"Inverse document frequency of 'trapper': {expected_idf:.2f}",
            stdout.getvalue(),
        )

    def test_get_bm25_idf_returns_expected_score(self) -> None:
        term_match_doc_count = len(self.inverted_index.get_documents("trapper"))
        expected = math.log(
            (len(self.inverted_index.docmap) - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

        self.assertAlmostEqual(self.inverted_index.get_bm25_idf("trapper"), expected)

    def test_get_bm25_idf_requires_single_token(self) -> None:
        with self.assertRaises(ValueError):
            self.inverted_index.get_bm25_idf("two terms")

    def test_bm25_idf_command_prints_bm25_idf_score(self) -> None:
        expected_bm25_idf = self.inverted_index.get_bm25_idf("trapper")

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            bm25_idf_command("trapper")

        self.assertIn(
            f"BM25 IDF score of 'trapper': {expected_bm25_idf:.2f}",
            stdout.getvalue(),
        )

    def test_get_bm25_tf_returns_expected_score(self) -> None:
        tf = self.inverted_index.get_tf(424, "trapper")
        doc_length = self.inverted_index.doc_lengths[424]
        avg_doc_length = sum(self.inverted_index.doc_lengths.values()) / len(
            self.inverted_index.doc_lengths
        )
        length_norm = 1 - BM25_B + BM25_B * (doc_length / avg_doc_length)
        expected = (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * length_norm)

        self.assertAlmostEqual(
            self.inverted_index.get_bm25_tf(424, "trapper"), expected
        )

    def test_bm25_returns_tf_times_idf(self) -> None:
        expected = self.inverted_index.get_bm25_tf(
            424, "trapper"
        ) * self.inverted_index.get_bm25_idf("trapper")

        self.assertAlmostEqual(self.inverted_index.bm25(424, "trapper"), expected)

    def test_bm25_search_returns_ranked_doc_scores(self) -> None:
        results = self.inverted_index.bm25_search("trapper", limit=3)

        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(doc_id, int) for doc_id, _ in results))
        self.assertTrue(all(isinstance(score, float) for _, score in results))
        self.assertGreaterEqual(results[0][1], results[1][1])
        self.assertGreaterEqual(results[1][1], results[2][1])

    def test_bm25_tf_command_uses_provided_k1(self) -> None:
        k1 = 3.0
        expected_bm25_tf = self.inverted_index.get_bm25_tf(424, "trapper", k1)

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            bm25_tf_command(424, "trapper", k1)

        self.assertIn(
            f"BM25 TF score of 'trapper' in document '424': {expected_bm25_tf:.2f}",
            stdout.getvalue(),
        )

    def test_bm25tf_cli_command_accepts_optional_k1(self) -> None:
        expected_bm25_tf = self.inverted_index.get_bm25_tf(424, "trapper", 3.0)
        result = subprocess.run(
            [
                sys.executable,
                "cli/keyword_search_cli.py",
                "bm25tf",
                "424",
                "trapper",
                "3.0",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn(
            f"BM25 TF score of 'trapper' in document '424': {expected_bm25_tf:.2f}",
            result.stdout,
        )

    def test_bm25search_cli_formats_title_and_rounded_score(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "cli/keyword_search_cli.py",
                "bm25search",
                "trapper",
                "--limit",
                "1",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertRegex(
            result.stdout.strip(),
            r"^1\. \(\d+\) .+ - Score: \d+\.\d{2}$",
        )

    def test_tfidf_command_prints_tfidf_score(self) -> None:
        expected_tfidf = self.inverted_index.get_tfidf(424, "trapper")

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            tfidf_command(424, "trapper")

        self.assertIn(
            f"TF-IDF score of 'trapper' in document '424': {expected_tfidf:.2f}",
            stdout.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
