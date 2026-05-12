import string

from .search_utils import DEFAULT_SEARCH_LIMIT, Movie, load_movies


PUNCT_TRANSLATION_TABLE: dict[int, int | None] = str.maketrans("", "", string.punctuation)
print("PUNCT_TRANSLATION_TABLE", PUNCT_TRANSLATION_TABLE)

def search_movies_by_title(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Movie]:
    movies = load_movies()
    normalized_query = preprocess_text(query)
    results: list[Movie] = []

    for movie in movies:
        normalized_title = preprocess_text(movie["title"])
        if normalized_query in normalized_title:
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def preprocess_text(text: str) -> str:
    return text.translate(PUNCT_TRANSLATION_TABLE).lower()
