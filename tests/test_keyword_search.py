import unittest

from cli.lib.keyword_search import (
    InvertedIndex,
    preprocess_text,
    search_movies_by_title,
    tokenize_text,
)
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT


class SearchMoviesByTitleTests(unittest.TestCase):
    def test_preprocess_text_lowercases_text(self) -> None:
        self.assertEqual(preprocess_text("GrEaT"), "great")

    def test_preprocess_text_removes_punctuation(self) -> None:
        self.assertEqual(
            preprocess_text("Faster, Pussycat! Kill! Kill!"),
            "faster pussycat kill kill",
        )

    def test_tokenize_text_splits_on_whitespace_and_discards_empty_tokens(self) -> None:
        self.assertEqual(
            tokenize_text("  great   bear\tadventure  "),
            ["great", "bear", "adventure"],
        )

    def test_returns_first_five_matches_in_dataset_order(self) -> None:
        results = search_movies_by_title("Great")

        self.assertEqual(len(results), DEFAULT_SEARCH_LIMIT)
        self.assertEqual(
            [movie["title"] for movie in results],
            [
                "The Land Before Time II: The Great Valley Adventure",
                "The First Great Train Robbery",
                "The Great Gatsby",
                "The Great Ziegfeld",
                "No Greater Love",
            ],
        )

    def test_search_is_case_insensitive(self) -> None:
        self.assertEqual(
            [movie["title"] for movie in search_movies_by_title("gReAt")],
            [movie["title"] for movie in search_movies_by_title("great")],
        )

    def test_matches_any_query_token_against_part_of_a_title_token(self) -> None:
        results = search_movies_by_title("furious fast")

        self.assertEqual(
            [movie["title"] for movie in results],
            [
                "Furious Seven",
                "Fast and Furious",
                "Faster, Pussycat! Kill! Kill!",
                "Furious 6",
                "Fast Times at Ridgemont High",
            ],
        )

    def test_removes_stopwords_from_query_before_matching(self) -> None:
        self.assertEqual(
            [movie["title"] for movie in search_movies_by_title("the hot shot")],
            [
                "Hot Potato",
                "Hot Shots! Part Deux",
                "Hotel Chevalier",
                "Hotel Berlin",
                "Killshot",
            ],
        )

    def test_stems_query_tokens_before_matching(self) -> None:
        self.assertEqual(
            [movie["title"] for movie in search_movies_by_title("running")],
            [
                "Virginia's Run",
                "Take the Money and Run",
                "Woman on the Run",
                "Honey, I Shrunk the Kids",
                "Runaway Train",
            ],
        )

    def test_build_inverted_index(self) -> None:
        inverted_index = InvertedIndex()

        inverted_index.build()

        docs = inverted_index.get_documents("merida")

        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0], 4651)
        self.assertEqual(inverted_index.docmap[4651]["title"], "Brave")


if __name__ == "__main__":
    unittest.main()
