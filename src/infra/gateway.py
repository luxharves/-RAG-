"""V7 — LLM gateway with retry, circuit breaker, provider failover and timeout.

Makes every LLM call resilient without touching call sites: ``LLMClient.chat``
is a facade that delegates to the shared ``get_gateway()`` singleton here.

Mechanism mirrors production LLM-gateway patterns (Tencent / APISIX):
  - per-attempt timeout (the SDK default is 600s — a hung API blocks a query)
  - exponential-backoff retry on 429 / 5xx / timeout / connection errors
  - per-provider circuit breaker (CLOSED → OPEN → HALF_OPEN probe)
  - multi-provider fallback chain (primary → backup_2 → backup_3 from .env)
  - canned graceful answer when every provider is down (contains
    "无法回答此问题" so VerifiedQA treats it as a self-refusal)
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import openai

DEFAULT_FALLBACK_ANSWER = "模型服务暂时不可用，无法回答此问题，请稍后重试。"


# ── Errors ──


class LLMGatewayError(Exception):
    """Base error for the gateway."""


class LLMUnavailableError(LLMGatewayError):
    """Raised when ALL providers fail and raise_on_total_failure is set."""


class RetryableProviderError(LLMGatewayError):
    """Internal: retry with backoff, then fail over to the next provider."""


class NonRetryableProviderError(LLMGatewayError):
    """Internal: fail over to the next provider immediately."""


# ── Config ──


@dataclass(frozen=True)
class CircuitConfig:
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base: float = 1.0
    multiplier: float = 2.0
    cap: float = 8.0
    jitter: bool = True
    sleep: Callable[[float], None] = time.sleep  # injectable for tests


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str

    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model) \
            and "replace-me" not in self.api_key \
            and "api.example.com" not in self.base_url


# ── Error classification ──


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def _classify_exc(exc: BaseException) -> bool:
    """Return True if the error is retryable.

    Order matters: RateLimit/InternalServer subclass APIStatusError.
    Timeout/connection subclass APIError (status None).
    """
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.InternalServerError):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and status >= 500
    if isinstance(exc, openai.OpenAIError):
        return False  # auth / bad request / etc.
    return True  # unknown (DNS/socket/httpx) → retryable


# ── Circuit breaker ──


class CircuitBreaker:
    """Per-provider state machine: CLOSED → OPEN → HALF_OPEN probe → ..."""

    CLOSED, OPEN, HALF_OPEN = "CLOSED", "OPEN", "HALF_OPEN"

    def __init__(self, config: CircuitConfig,
                 now: Callable[[], float] | None = None):
        self.config = config
        self._now = now or time.monotonic
        self._lock = threading.Lock()
        self.state = self.CLOSED
        self.failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == self.OPEN:
                if self._now() - self._opened_at >= self.config.cooldown_seconds:
                    self.state = self.HALF_OPEN
                    self._probe_in_flight = True
                    return True
                return False
            if self.state == self.HALF_OPEN:
                if not self._probe_in_flight:
                    self._probe_in_flight = True
                    return True
                return False
            return True  # CLOSED

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self._probe_in_flight = False
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            if self.state == self.HALF_OPEN:
                # probe failed → back to OPEN
                self.state = self.OPEN
                self._opened_at = self._now()
                self.failures = self.config.failure_threshold
            elif self.state == self.CLOSED:
                self.failures += 1
                if self.failures >= self.config.failure_threshold:
                    self.state = self.OPEN
                    self._opened_at = self._now()
            # OPEN: no-op

    def state_dict(self) -> dict:
        with self._lock:
            secs: float | None = None
            if self.state == self.OPEN and self._opened_at is not None:
                secs = max(0.0, self.config.cooldown_seconds - (self._now() - self._opened_at))
            return {
                "state": self.state,
                "consecutive_failures": self.failures,
                "opened_at": self._opened_at,
                "seconds_until_half_open": round(secs, 1) if secs is not None else None,
            }


# ── Provider (single attempt, OpenAI-backed) ──


class Provider:
    """One attempt against one endpoint. Raises typed errors on failure."""

    def __init__(self, cfg: ProviderConfig, timeout: float = 60.0):
        self.cfg = cfg
        self.timeout = timeout
        self._client: openai.OpenAI | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.cfg.name

    def _ensure_client(self) -> openai.OpenAI:
        with self._lock:
            if self._client is None:
                self._client = openai.OpenAI(
                    base_url=self.cfg.base_url,
                    api_key=self.cfg.api_key,
                    timeout=self.timeout,   # per-attempt cap
                    max_retries=0,          # gateway owns retries (no SDK double-retry)
                )
            return self._client

    def call(self, messages: list[dict], temperature: float = 0.0) -> tuple[str, dict]:
        client = self._ensure_client()
        try:
            response = client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 — classify, don't leak SDK types
            if _classify_exc(exc):
                raise RetryableProviderError(str(exc)) from exc
            raise NonRetryableProviderError(str(exc)) from exc
        content = response.choices[0].message.content or ""
        raw = {
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else None,
        }
        return content, raw


# ── Gateway ──


class LLMGateway:
    """Provider chain with retry, circuit breaker and graceful fallback."""

    def __init__(
        self,
        providers: list[Provider],
        *,
        retry_policy: RetryPolicy | None = None,
        circuit_config: CircuitConfig | None = None,
        fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
        raise_on_total_failure: bool = False,
        now: Callable[[], float] | None = None,
    ):
        self.providers = providers
        self.retry_policy = retry_policy or RetryPolicy()
        self.fallback_answer = fallback_answer
        self.raise_on_total_failure = raise_on_total_failure
        self._circuit_config = circuit_config or CircuitConfig()
        self._now = now  # clock injection for tests
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breakers_lock = threading.Lock()

    def _breaker(self, name: str) -> CircuitBreaker:
        with self._breakers_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(self._circuit_config, now=self._now)
            return self._breakers[name]

    def _backoff(self, attempt: int) -> float:
        delay = min(self.retry_policy.cap,
                    self.retry_policy.base * (self.retry_policy.multiplier ** attempt))
        if self.retry_policy.jitter:
            delay = random.uniform(0.5 * delay, delay)
        return delay

    def chat(self, messages: list[dict], temperature: float = 0.0) -> tuple[str, dict]:
        last_errors: list[dict] = []
        for provider in self.providers:
            breaker = self._breaker(provider.name)
            if not breaker.allow_request():
                last_errors.append({"provider": provider.name, "error": "circuit_open"})
                continue  # OPEN / probe in flight → next provider
            for attempt in range(self.retry_policy.max_retries + 1):
                try:
                    text, raw = provider.call(messages, temperature=temperature)
                except RetryableProviderError as exc:
                    last_errors.append({"provider": provider.name, "error": str(exc)})
                    if attempt < self.retry_policy.max_retries:
                        self.retry_policy.sleep(self._backoff(attempt))
                    else:
                        breaker.record_failure()  # retries exhausted → failover
                    continue
                except NonRetryableProviderError as exc:
                    last_errors.append({"provider": provider.name, "error": str(exc)})
                    breaker.record_failure()      # 4xx → failover immediately
                    break
                # success
                raw["provider"] = provider.name
                raw["attempts"] = attempt + 1
                breaker.record_success()
                return text, raw
        return self._total_failure(messages, last_errors)

    def _total_failure(self, messages: list[dict], errors: list[dict]) -> tuple[str, dict]:
        if self.raise_on_total_failure:
            raise LLMUnavailableError(f"all providers failed: {errors}")
        raw = {
            "model": None,
            "usage": None,
            "provider": "none",
            "gateway_fallback": True,
            "errors": errors,
            "circuit_state": self.state_dump(),
        }
        return self.fallback_answer, raw

    def state_dump(self) -> dict:
        return {
            "providers": [
                {"name": p.name, **self._breaker(p.name).state_dict()}
                for p in self.providers
            ]
        }


# ── Singleton (shared circuit state across LLMClient instances) ──


def _build_providers(settings) -> list[Provider]:
    cfgs = [
        ProviderConfig("primary", settings.llm_base_url,
                       settings.llm_api_key, settings.llm_model),
        ProviderConfig("backup_2", settings.llm_base_url_2,
                       settings.llm_api_key_2, settings.llm_model_2),
        ProviderConfig("backup_3", settings.llm_base_url_3,
                       settings.llm_api_key_3, settings.llm_model_3),
    ]
    timeout = settings.llm_timeout
    return [Provider(c, timeout=timeout) for c in cfgs if c.is_configured()]


@lru_cache(maxsize=1)
def get_gateway() -> LLMGateway:
    """Shared gateway built from settings; circuit state persists across calls.

    NOTE: reads settings once at first call (same caveat as deps.get_settings);
    call ``get_gateway.cache_clear()`` after editing .env at runtime.
    """
    from src.config.settings import settings
    return LLMGateway(
        providers=_build_providers(settings),
        retry_policy=RetryPolicy(
            max_retries=settings.llm_max_retries,
            base=settings.llm_retry_base,
            multiplier=settings.llm_retry_multiplier,
            cap=settings.llm_retry_cap,
        ),
        circuit_config=CircuitConfig(
            failure_threshold=settings.llm_circuit_threshold,
            cooldown_seconds=settings.llm_circuit_cooldown,
        ),
        fallback_answer=settings.llm_fallback_answer,
    )
