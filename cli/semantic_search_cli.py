#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
    semantic_search,
    chunk_text,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    subparsers.add_parser("verify", help="Verify the embedding model is loaded")

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    query_embed_parser = subparsers.add_parser(
        "embed_query", help="Generate an embedding for a single text"
    )
    query_embed_parser.add_argument("query", type=str, help="Query to embed")

    subparsers.add_parser("verify_embeddings", help="Verify movie embeddings")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of results to return",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of words in each chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Amount of words to overlap between chunks",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
