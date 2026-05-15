import argparse

from lib.keyword_search import search_command, build_command, tf_command, idf_command


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

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency of a term")
    idf_parser.add_argument("term", type=str)

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
