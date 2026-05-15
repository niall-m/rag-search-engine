import math
import os
import pickle
import string

from nltk.stem import PorterStemmer
from collections import Counter

from .search_utils import (
    CACHE_DIR,
    INDEX_DISK_DATA_PATH,
    DOCMAP_DISK_DATA_PATH,
    TF_DISK_DATA_PATH,
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
        self.term_frequencies: dict[int, Counter] = {}

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
        with open(TF_DISK_DATA_PATH, "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self) -> None:
        if (
            not os.path.exists(INDEX_DISK_DATA_PATH)
            or not os.path.exists(DOCMAP_DISK_DATA_PATH)
            or not os.path.exists(TF_DISK_DATA_PATH)
        ):
            raise FileNotFoundError

        with open(INDEX_DISK_DATA_PATH, "rb") as file:
            self.index = pickle.load(file)
        with open(DOCMAP_DISK_DATA_PATH, "rb") as file:
            self.docmap = pickle.load(file)
        with open(TF_DISK_DATA_PATH, "rb") as file:
            self.term_frequencies = pickle.load(file)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) == 0:
            return 0
        if len(tokens) > 1:
            raise ValueError("Only one term token at a time, please")
        token = tokens[0]
        if (
            doc_id not in self.term_frequencies
            or token not in self.term_frequencies[doc_id]
        ):
            return 0
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(token, set()))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies.setdefault(doc_id, Counter())
        for token in set(tokens):
            self.index.setdefault(token, set()).add(doc_id)
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1


def build_command() -> None:
    print("Building inverted index...")
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    print(inverted_index)


def tf_command(doc_id, term) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()
    frequency = inverted_index.get_tf(doc_id, term)
    print(f"Frequency of '{term}': {frequency}")


def idf_command(term: str) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()
    idf = inverted_index.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def search_command(
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
