import string

from .search_utils import DEFAULT_SEARCH_LIMIT, Movie, load_movies


PUNCT_TRANSLATION_TABLE: dict[int, int | None] = str.maketrans("", "", string.punctuation)


def search_movies_by_title(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Movie]:
    movies = load_movies()
    query_tokens = tokenize_text(query)
    results: list[Movie] = []

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> str:
    return text.translate(PUNCT_TRANSLATION_TABLE).lower()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    return [token for token in text.split() if token]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
