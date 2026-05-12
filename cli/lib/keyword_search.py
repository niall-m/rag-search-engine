from .search_utils import DEFAULT_SEARCH_LIMIT, Movie, load_movies


def search_movies_by_title(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Movie]:
    movies = load_movies()
    results: list[Movie] = []
    for movie in movies:
        p_query = preprocess_text(query)
        p_title = preprocess_text(movie["title"])
        if p_query in p_title:
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> str:
    return text.lower()
