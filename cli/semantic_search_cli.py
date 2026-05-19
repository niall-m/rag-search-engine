#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )

    subparsers.add_parser("verify", help="Verify the embedding model is loaded")

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)

if __name__ == "__main__":
    main()
