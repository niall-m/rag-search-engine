import argparse

from lib.keyword_search import search_movies_by_title


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    search_parser = subparsers.add_parser("search", help="Search movies by title")
    search_parser.add_argument("query", type=str, help="Search query")
    return parser


def run_search_command(query: str) -> None:
    print(f"Searching for: {query}")
    results = search_movies_by_title(query)
    for index, movie in enumerate(results, start=1):
        print(f"{index}. {movie['title']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "search":
            run_search_command(args.query)

if __name__ == "__main__":
    main()
