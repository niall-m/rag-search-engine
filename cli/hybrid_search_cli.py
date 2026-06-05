import argparse
import logging

from lib.hybrid_search import (
    RRF_DEBUG_LOGGER_NAME,
    normalize_command,
    weighted_search_command,
    rrf_search_command,
)
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
        choices=["individual", "batch", "cross_encoder"],
        help="Re-rank method",
    )
    rrf_search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logging for each RRF pipeline stage",
    )

    return parser


def configure_debug_logging() -> None:
    logger = logging.getLogger(RRF_DEBUG_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


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
            if args.debug:
                configure_debug_logging()

            result = rrf_search_command(
                args.query,
                args.k,
                args.enhance,
                args.rerank_method,
                args.limit,
                args.debug,
            )

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): "
                    f"'{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            if result["reranked"]:
                print(
                    f"Re-ranking top {len(result['results'])} results using "
                    f"{result['rerank_method']} method...\n"
                )

            print(
                f"Reciprocal Rank Fusion Results for "
                f"'{result['query']}' (k={result['k']}):\n"
            )

            for index, search_result in enumerate(result["results"], start=1):
                print(f"{index}. {search_result['title']}")
                if "rerank_score" in search_result:
                    print(f"  Re-rank Score: {search_result['rerank_score']:.3f}/10")
                if "rerank_rank" in search_result:
                    print(f"  Re-rank Rank: {search_result['rerank_rank']}")
                if "rerank_cross_score" in search_result:
                    print(
                        f"  Cross Encoder Score: {search_result['rerank_cross_score']:.3f}"
                    )
                print(f"  RRF Score: {search_result['rrf_score']:.3f}")
                print(
                    "  "
                    f"BM25 Rank: {search_result['bm25_rank']}, "
                    f"Semantic Rank: {search_result['semantic_rank']}"
                )
                print(f"  {search_result['description'][:DOCUMENT_PREVIEW_LENGTH]}...")
                print()
        case _:
            raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
