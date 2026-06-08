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

def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search = HybridSearch(load_movies())
    search_results = search.rrf_search(query, limit=limit)
    client = create_client()
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{search_results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    response_text = (response.text or "").strip()
    return {
        "search_results": search_results,
        "llm_response": response_text,
    }


def citation_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search = HybridSearch(load_movies())
    search_results = search.rrf_search(query, limit=limit)
    client = create_client()
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{search_results}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    response = client.models.generate_content(model=MODEL, contents=prompt)
    response_text = (response.text or "").strip()
    return {
        "search_results": search_results,
        "llm_response": response_text,
    }
