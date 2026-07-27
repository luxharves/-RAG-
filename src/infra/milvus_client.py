"""Milvus adapter for Phase 0 smoke testing.

Uses the MilvusClient API (pymilvus >= 3.0) which supports both
Milvus Lite (local file) and Milvus Server via configuration.
"""

from pymilvus import MilvusClient
from src.config.settings import settings


class MilvusAdapter:
    """Minimal Milvus adapter for connectivity verification."""

    def __init__(self, uri: str | None = None, token: str | None = None):
        self.uri = uri or settings.milvus_uri
        self.token = token or settings.milvus_token
        self._client: MilvusClient | None = None

    def connect(self) -> None:
        """Establish connection to Milvus (Lite or Server)."""
        if self.uri.startswith("http"):
            self._client = MilvusClient(uri=self.uri, token=self.token or None)
        else:
            # Milvus Lite — local file path
            self._client = MilvusClient(self.uri)

    def disconnect(self) -> None:
        """Close the Milvus client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def create_collection(self, name: str, dim: int) -> None:
        """Create a temporary collection with a float vector field."""
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        if self._client.has_collection(name):
            self._client.drop_collection(name)

        self._client.create_collection(
            collection_name=name,
            dimension=dim,
            metric_type="COSINE",
            auto_id=True,
            # Enable varchar field for text
            # (MilvusClient.create_collection with dimension auto-creates id, vector)
        )

    def insert(self, collection_name: str, text: str, vector: list[float]) -> None:
        """Insert a single row into the collection."""
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        self._client.insert(
            collection_name=collection_name,
            data=[{"text": text, "vector": vector}],
        )

    def search(
        self, collection_name: str, vector: list[float], top_k: int = 1
    ) -> list[dict]:
        """Search for nearest neighbors. Returns list of result dicts."""
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        results = self._client.search(
            collection_name=collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["text"],
        )
        return [
            {"id": hit["id"], "distance": hit["distance"], "text": hit["entity"].get("text", "")}
            for hit in results[0]
        ]

    def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""
        if self._client is not None and self._client.has_collection(name):
            self._client.drop_collection(name)

    @property
    def is_connected(self) -> bool:
        return self._client is not None
