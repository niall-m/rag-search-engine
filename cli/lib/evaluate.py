import json
from .search_utils import RRFSearchCommandResult
from .query_enhancement import MODEL, create_client


def llm_judge_results(result: RRFSearchCommandResult) -> list[int]:
    client = create_client()

    query = result["original_query"]
    formatted_results: list[str] = []
    for index, search_result in enumerate(result["results"], start=1):
        formatted_results.append(f"{index}. {search_result['title']}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    response_text = (response.text or "").strip()
    scores = json.loads(response_text)

    if len(scores) != len(result["results"]):
        raise ValueError("LLM evaluation response length did not match result count")

    return [int(score) for score in scores]


def evaluate(result: RRFSearchCommandResult) -> None:
    scores = llm_judge_results(result)

    for index, (search_result, score) in enumerate(zip(result["results"], scores), 1):
        if score not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid LLM evaluation score: {score}")
        print(f"{index}. {search_result['title']}: {score}/3")
