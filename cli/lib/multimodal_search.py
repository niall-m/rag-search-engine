import os

from typing import Any
from PIL import Image
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> NDArray[Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        embeddings = self.model.encode([image])

        return embeddings[0]


def verify_image_embedding_command(image_path: str):
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
