"""BGE-Reranker adapter.

Loads BGE-Reranker locally and exposes a minimal scoring interface for smoke testing.
"""

from sentence_transformers import CrossEncoder
from src.config.settings import settings


class Reranker:
    """Thin wrapper around CrossEncoder for BGE-Reranker scoring."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        model_name = model_name or settings.reranker_model
        device = device or settings.model_device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self._model: CrossEncoder | None = None

    def load(self) -> None:
        """Load the model into memory."""
        self._model = CrossEncoder(self.model_name, device=self.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score each document against the query. Higher is better."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        return scores.tolist() if hasattr(scores, "tolist") else list(scores)
