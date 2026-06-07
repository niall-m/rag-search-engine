import argparse

from lib.augmented_generation import rag_search_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    return parser


def run_command(args: argparse.Namespace) -> None:
    match args.command:
        case "rag":
            results = rag_search_command(args.query)
            print("Search Results:")
            for res in results["search_results"]:
                print(f"- {res['title']}")
            print()
            print("RAG Response:")
            print(results["llm_response"])
        case _:
            raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
