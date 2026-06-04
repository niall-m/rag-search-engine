import argparse

from lib.hybrid_search import (
    normalize_command,
    weighted_search_command,
    rrf_search_command,
)
from lib.query_enhancement import enhance_query
from lib.search_utils import (
    DEFAULT_K,
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

    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="Search movies with reciprocal rank fusion",
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help="Reciprocal rank weighting for higher vs lower hybrid score",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results to return",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch"],
        help="Re-rank method",
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
        case "rrf-search":
            query = args.query
            if args.enhance:
                enhanced_query = enhance_query(query, args.enhance)
                print(
                    f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query

            if args.rerank_method:
                print(
                    f"Re-ranking top {args.limit} results using {args.rerank_method} method...\n"
                )

            results = rrf_search_command(query, args.k, args.limit, args.rerank_method)

            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")

            for index, result in enumerate(results[: args.limit], start=1):
                print(f"{index}. {result['title']}")
                if args.rerank_method == "individual" and "rerank_score" in result:
                    print(f"  Re-rank Score: {result['rerank_score']:.3f}/10")
                if args.rerank_method == "batch" and "rerank_rank" in result:
                    print(f"  Re-rank Rank: {result['rerank_rank']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(
                    "  "
                    f"BM25 Rank: {result['bm25_rank']}, "
                    f"Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {result['description'][:DOCUMENT_PREVIEW_LENGTH]}...")
                print()
        case _:
            raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
