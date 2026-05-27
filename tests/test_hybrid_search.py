import unittest
from io import StringIO
from unittest.mock import patch

from cli.lib.hybrid_search import normalize_command


class NormalizeCommandTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
