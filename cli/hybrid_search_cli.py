import argparse

from lib.hybrid_search import normalize_command, weighted_search_command
from lib.search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize some numbers",
    )
    normalize_parser.add_argument(
        "nums",
        type=float,
        nargs="*",
        help="Pass a list of numbers to normalize",
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Search movies with weighted hybrid search",
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Keyword weighting for the hybrid score",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results to return",
    )

    return parser


def run_command(args: argparse.Namespace) -> None:
    match args.command:
        case "normalize":
            normalize_command(args.nums)
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for index, result in enumerate(results[: args.limit], start=1):
                print(f"{index}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    "  "
                    f"BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(f"  {result['description'][:DOCUMENT_PREVIEW_LENGTH]}...")
        case _:
            raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
