import unittest

from cli.lib.keyword_search import search_movies_by_title
from cli.lib.search_utils import DEFAULT_SEARCH_LIMIT


class SearchMoviesByTitleTests(unittest.TestCase):
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

    def test_returns_all_matches_when_under_limit(self) -> None:
        results = search_movies_by_title("Great Valley Adventure")

        self.assertEqual(
            [movie["title"] for movie in results],
            ["The Land Before Time II: The Great Valley Adventure"],
        )


if __name__ == "__main__":
    unittest.main()
