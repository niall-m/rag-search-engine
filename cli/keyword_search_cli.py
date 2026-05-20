import argparse

from lib.keyword_search import (
    BM25_K1,
    BM25_B,
    InvertedIndex,
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    subparsers.add_parser("build", help="Build the InvertedIndex")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency in target doc")
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency of a term"
    )
    idf_parser.add_argument("term", type=str)

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get term frequency, inverse document frequency in target doc"
    )
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str)

    bm25tf_parser = subparsers.add_parser(
        "bm25tf", help="Get term frequency, inverse document frequency in target doc"
    )
    bm25tf_parser.add_argument("doc_id", type=int)
    bm25tf_parser.add_argument("term", type=str)
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1)
    bm25tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B)

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results to return",
    )

    search_parser = subparsers.add_parser("search", help="Search movies by title")
    search_parser.add_argument("query", type=str, help="Search query")

    return parser


def run_command(args: argparse.Namespace) -> None:
    match args.command:
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25_idf_command(args.term)
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
        case "bm25search":
            try:
                inverted_index = InvertedIndex()
                inverted_index.load()
            except FileNotFoundError:
                print("Search index not found. Run the build command first.")
                return

            results = inverted_index.bm25_search(args.query, args.limit)
            for index, (doc_id, score) in enumerate(results, start=1):
                movie = inverted_index.docmap[doc_id]
                print(f"{index}. ({doc_id}) {movie['title']} - Score: {score:.2f}")
        case "search":
            print(f"Searching for: {args.query}")
            try:
                results = search_command(args.query)
            except FileNotFoundError:
                print("Search index not found. Run the build command first.")
                return

            for index, movie in enumerate(results, start=1):
                print(f"{index}. {movie['title']} - ID: {movie['id']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
