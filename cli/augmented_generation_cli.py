import argparse

from lib.augmented_generation import rag_search_command, summarize_command, citation_command, question_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT


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

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate multi-document summary"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of documents to summarize",
    )

    citation_parser = subparsers.add_parser(
        "citations", help="Generate summary with citations"
    )
    citation_parser.add_argument("query", type=str, help="Search query for summarization and citation")
    citation_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of documents to summarize with citation",
    )

    question_parser = subparsers.add_parser(
        "question", help="Generate answer to a question"
    )
    question_parser.add_argument("question", type=str, help="Question to be answered by LLM")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of documents to summarize with citation",
    )

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
        case "summarize":
            results = summarize_command(args.query, args.limit)
            print("Search Results:")
            for res in results["search_results"]:
                print(f"- {res['title']}")
            print()
            print("RAG Response:")
            print(results["llm_response"])
        case "citations":
            results = citation_command(args.query, args.limit)
            print("Search Results:")
            for res in results["search_results"]:
                print(f"- {res['title']}")
            print()
            print("RAG Response:")
            print(results["llm_response"])
        case "question":
            results = question_command(args.question, args.limit)
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
