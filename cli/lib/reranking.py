import json
import re
import time

from typing import Literal
from google import genai
from sentence_transformers import CrossEncoder

from .query_enhancement import MODEL, create_client
from .search_utils import (
    HybridRankResult,
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    INDIVIDUAL_RERANK_DELAY_SECONDS,
    RERANK_RESULT_MULTIPLIER,
)

RERANK_SCORE_PATTERN = re.compile(r"10(?:\.0+)?|[0-9](?:\.\d+)?")


def individual_rerank_score(
    query: str,
    title: str,
    document: str,
    client: genai.Client | None = None,
) -> float:
    active_client = client or create_client()
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


def rerank_individually(
    query: str,
    results: list[HybridRankResult],
) -> list[HybridRankResult]:
    client = create_client()

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


def rerank_batch(
    query: str,
    results: list[HybridRankResult],
) -> list[HybridRankResult]:
    client = create_client()
    docs: list[str] = []
    for result in results:
        docs.append(
            f"{result['id']}: {result['title']} - "
            f"{result['description'][:DOCUMENT_PREVIEW_LENGTH]}"
        )
    doc_list_str = "\n".join(docs)
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    response_text = (response.text or "").strip()
    ranked_ids = json.loads(response_text)
    result_map = {result["id"]: result for result in results}

    next_rank = 1
    for movie_id in ranked_ids:
        result = result_map.get(movie_id)
        if result is None:
            continue
        result["rerank_rank"] = next_rank
        next_rank += 1

    for result in results:
        if "rerank_rank" not in result:
            result["rerank_rank"] = next_rank
            next_rank += 1

    return sorted(
        results,
        key=lambda result: result["rerank_rank"],
    )


def rerank_cross_encoder(
    query: str, results: list[HybridRankResult]
) -> list[HybridRankResult]:
    try:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    except Exception:
        cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu"
        )

    pairs: list[list[str]] = []

    for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])

    scores = cross_encoder.predict(pairs)
    for result, score in zip(results, scores):
        result["rerank_cross_score"] = score

    return sorted(
        results, key=lambda result: result["rerank_cross_score"], reverse=True
    )


def rerank(
    query: str,
    results: list[HybridRankResult],
    method: Literal["individual", "batch", "cross_encoder"] | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[HybridRankResult]:
    rerank_limit = limit * RERANK_RESULT_MULTIPLIER

    if method == "individual":
        reranked_results = rerank_individually(query, results[:rerank_limit])
        return reranked_results[:limit]

    if method == "batch":
        reranked_results = rerank_batch(query, results[:rerank_limit])
        return reranked_results[:limit]

    if method == "cross_encoder":
        reranked_results = rerank_cross_encoder(query, results[:rerank_limit])
        return reranked_results[:limit]

    return results[:limit]
