import os
import pickle
import string

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    INDEX_DISK_DATA_PATH,
    DOCMAP_DISK_DATA_PATH,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    load_movies,
    load_stopwords,
)


PUNCT_TRANSLATION_TABLE: dict[int, int | None] = str.maketrans(
    "", "", string.punctuation
)


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}

    def __repr__(self) -> str:
        return f"InvertedIndex(tokens={len(self.index)}, documents={len(self.docmap)})"

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            input_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], input_text)
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_DISK_DATA_PATH, "wb") as file:
            pickle.dump(self.index, file)
        with open(DOCMAP_DISK_DATA_PATH, "wb") as file:
            pickle.dump(self.docmap, file)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)

    def __add_document(self, doc_id: int, text: str) -> None:
        text_tokens = tokenize_text(text)
        for token in set(text_tokens):
            self.index.setdefault(token, set()).add(doc_id)


def build_inverted_index() -> None:
    print("Building inverted index...")
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    print(inverted_index)
    print(
        f"First document for token 'merida' = {inverted_index.get_documents('merida')[0]}"
    )


def search_movies_by_title(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Movie]:
    movies = load_movies()
    stopwords = load_stopwords()
    query_tokens = tokenize_text(query)
    filtered_query_tokens = [t for t in query_tokens if t not in stopwords]
    stemmer = PorterStemmer()
    stemmed_query_tokens = stem_tokens(filtered_query_tokens, stemmer)

    results: list[Movie] = []

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        filtered_title_tokens = [t for t in title_tokens if t not in stopwords]
        stemmed_title_tokens = stem_tokens(filtered_title_tokens, stemmer)
        if has_matching_token(stemmed_query_tokens, stemmed_title_tokens):
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


def stem_tokens(tokens: list[str], stemmer: PorterStemmer) -> list[str]:
    return [stemmer.stem(token) for token in tokens]
