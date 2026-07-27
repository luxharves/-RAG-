"""BGE-M3 embedding adapter.

Loads BGE-M3 locally and exposes a minimal encode interface for smoke testing.
"""

from sentence_transformers import SentenceTransformer
from src.config.settings import settings


class Embedder:
    """Thin wrapper around SentenceTransformer for BGE-M3 embeddings."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        model_name = model_name or settings.embedding_model
        device = device or settings.model_device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        """Load the model into memory."""
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(self, text: str) -> list[float]:
        """Encode a single text into a float vector."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_batch(
        self, texts: list[str], batch_size: int = 8
    ) -> list[list[float]]:
        """Encode a batch of texts into float vectors."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model.get_embedding_dimension()
