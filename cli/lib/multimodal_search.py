import os

from typing import Any
from PIL import Image
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .search_utils import Movie, MultiModalResult, load_movies
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(
        self,
        documents: list[Movie] | None = None,
        model_name: str = "clip-ViT-B-32",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else load_movies()
        self.texts: list[str] = self.load_texts()
        self.text_embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True,
        )

    def embed_image(self, image_path: str) -> NDArray[Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        embeddings = self.model.encode([image], show_progress_bar=True)

        return embeddings[0]

    def load_texts(self) -> list[str]:
        return [
            f"{document['title']}: {document['description']}"
            for document in self.documents
        ]

    def search_with_image(self, image_path: str) -> list[MultiModalResult]:
        embedding = self.embed_image(image_path)
        results: list[MultiModalResult] = []

        for text_embedding, document in zip(self.text_embeddings, self.documents):
            score = cosine_similarity(embedding, text_embedding)
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "description": document["description"],
                    "similarity_score": score,
                }
            )

        results = sorted(
            results,
            key=lambda result: result["similarity_score"],
            reverse=True,
        )
        return results[:5]


def verify_image_embedding_command(image_path: str):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    multimodal_search = MultimodalSearch()
    return multimodal_search.search_with_image(image_path)
