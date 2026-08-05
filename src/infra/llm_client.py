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
        self._override = (base_url, api_key, model)
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self._client: OpenAI | None = None
        self._single_gateway = None

    def _single_gw(self):
        """Private gateway for explicit-arg instances (isolated circuit state)."""
        from src.infra.gateway import LLMGateway, Provider, ProviderConfig
        if self._single_gateway is None:
            cfg = ProviderConfig("custom", self.base_url, self.api_key, self.model)
            self._single_gateway = LLMGateway(providers=[Provider(cfg)])
        return self._single_gateway

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

    def chat(self, messages: list[dict], temperature: float = 0.0) -> tuple[str, dict]:
        """Send a chat completion request through the V7 LLM gateway.

        No-arg instances route to the shared gateway (retry / circuit breaker /
        provider failover / timeout). Explicit-arg instances use a private
        single-provider gateway so their semantics are preserved.

        Returns (response_text, raw_response_dict).
        """
        if any(x is not None for x in self._override):
            return self._single_gw().chat(messages, temperature=temperature)
        from src.infra.gateway import get_gateway
        return get_gateway().chat(messages, temperature=temperature)
