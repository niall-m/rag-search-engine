import argparse
import json
from pathlib import Path
from typing import Protocol, TypedDict

from lib.hybrid_search import HybridSearch
from lib.search_utils import (
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    GOLDEN_PATH,
    HybridRankResult,
    load_movies,
)


class GoldenTestCase(TypedDict):
    query: str
    relevant_docs: list[str]


class EvaluationResult(TypedDict):
    query: str
    precision: float
    retrieved_titles: list[str]
    relevant_titles: list[str]


class RRFEvaluator(Protocol):
    def rrf_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[HybridRankResult]: ...


def load_golden_test_cases(path: Path = GOLDEN_PATH) -> list[GoldenTestCase]:
    with path.open("r", encoding="utf-8") as golden_file:
        data = json.load(golden_file)
    return data["test_cases"]


def calculate_precision_at_k(
    retrieved_titles: list[str],
    relevant_titles: list[str],
) -> float:
    if not retrieved_titles:
        return 0.0

    relevant_retrieved = [
        title for title in retrieved_titles if title in relevant_titles
    ]
    return len(relevant_retrieved) / len(retrieved_titles)


def evaluate_precision_at_k(
    search: RRFEvaluator,
    test_cases: list[GoldenTestCase],
    limit: int,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []

    for test_case in test_cases:
        retrieved_results = search.rrf_search(test_case["query"], DEFAULT_K, limit)
        retrieved_titles = [str(result["title"]) for result in retrieved_results]

        results.append(
            {
                "query": test_case["query"],
                "precision": calculate_precision_at_k(
                    retrieved_titles,
                    test_case["relevant_docs"],
                ),
                "retrieved_titles": retrieved_titles,
                "relevant_titles": test_case["relevant_docs"],
            }
        )

    return results


def format_titles(titles: list[str]) -> str:
    return ", ".join(titles)


def print_evaluation_results(results: list[EvaluationResult], limit: int) -> None:
    print(f"k={limit}")
    print()

    for result in results:
        print(f"- Query: {result['query']}")
        print(f"  - Precision@{limit}: {result['precision']:.4f}")
        print(f"  - Retrieved: {format_titles(result['retrieved_titles'])}")
        print(f"  - Relevant: {format_titles(result['relevant_titles'])}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    if limit < 1:
        parser.error("--limit must be greater than 0")

    search = HybridSearch(load_movies())
    test_cases = load_golden_test_cases()
    results = evaluate_precision_at_k(search, test_cases, limit)
    print_evaluation_results(results, limit)


if __name__ == "__main__":
    main()
