"""OpenAI-compatible LLM adapter for Phase 0 smoke testing."""

from openai import OpenAI
from src.config.settings import settings


class LLMClient:
    """Minimal OpenAI-compatible chat completion client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self._client: OpenAI | None = None

    @property
    def is_configured(self) -> bool:
        """Check whether the client has non-placeholder configuration."""
        return (
            self.api_key != "replace-me"
            and self.base_url != "https://api.example.com/v1"
            and self.model != "replace-me"
        )

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Send a chat completion request.

        Returns (response_text, raw_response_dict).
        """
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        raw = {"model": response.model, "usage": response.usage.model_dump() if response.usage else None}
        return content, raw
