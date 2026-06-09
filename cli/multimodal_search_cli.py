import argparse

from lib.multimodal_search import verify_image_embedding_command


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

    return parser


def run_command(args: argparse.Namespace) -> None:
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding_command(args.image_path)
        case _:
            raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
