from .search_utils import DEFAULT_SEARCH_LIMIT, Movie, load_movies


def search_movies_by_title(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Movie]:
    movies = load_movies()
    results: list[Movie] = []
    for movie in movies:
        if query in movie["title"]:
            results.append(movie)
            if len(results) >= limit:
                break
    return results
