from typing import TypedDict

from .hybrid_search import HybridSearch
from .query_enhancement import MODEL, create_client
from .search_utils import DEFAULT_SEARCH_LIMIT, HybridRankResult, load_movies


class RAGSearchCommandResult(TypedDict):
    search_results: list[HybridRankResult]
    llm_response: str


def format_documents_for_prompt(results: list[HybridRankResult]) -> str:
    if not results:
        return "(no documents retrieved)"

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result['title']}\n"
            f"Description: {result['description']}"
        )
    return "\n\n".join(lines)


def rag_search_command(query: str) -> RAGSearchCommandResult:
    search = HybridSearch(load_movies())
    search_results = search.rrf_search(query, limit=DEFAULT_SEARCH_LIMIT)
    client = create_client()
    docs = format_documents_for_prompt(search_results)
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.
Use only the retrieved documents below.
If the documents are insufficient to fully answer the query, say so.
Mention movie titles explicitly when you reference them.

Query: {query}

Documents:
{docs}

Answer:"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    response_text = (response.text or "").strip()
    return {
        "search_results": search_results,
        "llm_response": response_text,
    }
