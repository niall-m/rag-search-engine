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

    def load(self) -> None:
        if not os.path.exists(INDEX_DISK_DATA_PATH) or not os.path.exists(
            DOCMAP_DISK_DATA_PATH
        ):
            raise FileNotFoundError

        with open(INDEX_DISK_DATA_PATH, "rb") as file:
            self.index = pickle.load(file)

        with open(DOCMAP_DISK_DATA_PATH, "rb") as file:
            self.docmap = pickle.load(file)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index.setdefault(token, set()).add(doc_id)


def build_inverted_index() -> None:
    print("Building inverted index...")
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    print(inverted_index)


def search_movies_by_title(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[Movie]:
    inverted_index = InvertedIndex()
    inverted_index.load()

    seen_doc_ids: set[int] = set()
    results: list[Movie] = []

    query_tokens = tokenize_text(query)

    for token in query_tokens:
        matching_doc_ids = inverted_index.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen_doc_ids:
                continue

            results.append(inverted_index.docmap[doc_id])
            seen_doc_ids.add(doc_id)
            if len(results) >= limit:
                return results

    return results


def preprocess_text(text: str) -> str:
    return text.translate(PUNCT_TRANSLATION_TABLE).lower()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    tokens = [token for token in text.split() if token]

    filtered_tokens = [t for t in tokens if t not in stopwords]

    return [stemmer.stem(token) for token in filtered_tokens]
