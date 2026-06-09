import argparse

from lib.multimodal_search import verify_image_embedding_command, image_search_command
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify image embeddings")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    verify_image_embeddings_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify image embeddings",
    )
    verify_image_embeddings_parser.add_argument(
        "image_path",
        help="Path to an image file",
    )

    image_search_parser = subparsers.add_parser(
        "image_search",
        help="Search movies vs image simularity",
    )
    image_search_parser.add_argument(
        "image_path",
        help="Path to an image file",
    )

    return parser


def run_command(args: argparse.Namespace) -> None:
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding_command(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for index, search_result in enumerate(results, start=1):
                preview = search_result["description"][:DOCUMENT_PREVIEW_LENGTH]
                if len(search_result["description"]) > DOCUMENT_PREVIEW_LENGTH:
                    preview += "..."

                print(
                    f"{index}. {search_result['title']} "
                    f"(similarity: {search_result['similarity_score']:.3f})"
                )
                print(f"   {preview}\n")
        case _:
            raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
