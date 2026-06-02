from functools import lru_cache
import os
import re
import time
from typing import Literal

from dotenv import load_dotenv
from google import genai
from .search_utils import HybridRankResult, INDIVIDUAL_RERANK_DELAY_SECONDS

MODEL = "gemma-4-31b-it"
RERANK_SCORE_PATTERN = re.compile(r"10(?:\.0+)?|[0-9](?:\.\d+)?")


@lru_cache(maxsize=1)
def _create_client() -> genai.Client:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def spell_correct(query: str) -> str:
    client = _create_client()
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def rewrite_query(query: str) -> str:
    client = _create_client()
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    rewritten = (response.text or "").strip().strip('"')
    return rewritten if rewritten else query


def expand_query(query: str) -> str:
    client = _create_client()
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    expanded = (response.text or "").strip().strip('"')
    return expanded if expanded else query


def enhance_query(query: str, method: Literal["spell", "rewrite", "expand"]) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)


def individual_rerank_score(
    query: str,
    title: str,
    document: str,
    client: genai.Client | None = None,
) -> float:
    active_client = client or _create_client()
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {document}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    response = active_client.models.generate_content(model=MODEL, contents=prompt)
    response_text = (response.text or "").strip()
    score_match = RERANK_SCORE_PATTERN.search(response_text)
    if score_match is None:
        return 0.0

    score = float(score_match.group())
    return min(max(score, 0.0), 10.0)


def rerank_results_individually(
    query: str,
    results: list[HybridRankResult],
) -> list[HybridRankResult]:
    client = _create_client()

    for index, result in enumerate(results):
        result["rerank_score"] = individual_rerank_score(
            query,
            result["title"],
            result["description"],
            client,
        )
        if index < len(results) - 1:
            time.sleep(INDIVIDUAL_RERANK_DELAY_SECONDS)

    return sorted(
        results,
        key=lambda result: result["rerank_score"],
        reverse=True,
    )
