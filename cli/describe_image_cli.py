import argparse
import mimetypes
import os

from google.genai import types
from lib.query_enhancement import MODEL, create_client


system_prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image + text → rewritten query")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--query", required=True, help="A text query to rewrite based on the image")

    return parser


def run_command(args: argparse.Namespace) -> None:
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        img = f.read()

    client = create_client()
    parts = [
        system_prompt.strip(),
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    response = client.models.generate_content(model=MODEL, contents=parts)
    if response.text is None:
        raise RuntimeError("No text in Gemini API response")

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
