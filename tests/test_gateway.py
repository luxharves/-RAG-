"""V7 gateway tests — deterministic, NO real API.

Uses FakeProvider (exposes .name + .call) and an injectable sleep + clock.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infra.gateway import (  # noqa: E402
    CircuitBreaker, CircuitConfig, LLMGateway, LLMUnavailableError,
    NonRetryableProviderError, RetryPolicy, RetryableProviderError,
    _build_providers, _classify_exc, _is_retryable_status,
)


# ── Helpers ──


class FakeProvider:
    """Minimal provider: .name + .call; records invocations."""

    def __init__(self, name, call_fn):
        self.name = name
        self._call = call_fn
        self.calls = 0

    def call(self, messages, temperature=0.0):
        self.calls += 1
        return self._call(messages, temperature)


class FakeClock:
    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _retry(max_retries=2, sleep_rec=None):
    return RetryPolicy(max_retries=max_retries, base=1.0, multiplier=2.0,
                       cap=8.0, jitter=False, sleep=(sleep_rec or (lambda d: None)))


def _ok(text="ok", model="m"):
    return text, {"model": model, "usage": None}


def _gateway(providers, **kw):
    kw.setdefault("retry_policy", _retry())
    kw.setdefault("circuit_config", CircuitConfig(failure_threshold=3, cooldown_seconds=30))
    return LLMGateway(providers, **kw)


# ── Retry policy ──


class TestRetry:
    def test_retry_on_500(self):
        state = {"n": 0}
        def fn(messages, temperature):
            state["n"] += 1
            if state["n"] < 3:
                raise RetryableProviderError("500")
            return _ok()
        p = FakeProvider("primary", fn)
        gw = _gateway([p])
        text, raw = gw.chat([{"role": "user", "content": "hi"}])
        assert text == "ok"
        assert state["n"] == 3
        assert raw["provider"] == "primary"
        assert raw["attempts"] == 3
        assert gw._breaker("primary").state == CircuitBreaker.CLOSED

    def test_no_retry_on_400(self):
        def bad(messages, temperature):
            raise NonRetryableProviderError("400")
        def good(messages, temperature):
            return _ok("backup_ok", "m2")
        p1 = FakeProvider("primary", bad)
        p2 = FakeProvider("backup", good)
        gw = _gateway([p1, p2])
        text, raw = gw.chat([{"role": "user", "content": "hi"}])
        assert text == "backup_ok"
        assert p1.calls == 1          # no retry on non-retryable
        assert raw["provider"] == "backup"

    def test_retry_on_429(self):
        state = {"n": 0}
        def fn(messages, temperature):
            state["n"] += 1
            if state["n"] == 1:
                raise RetryableProviderError("429")
            return _ok()
        p = FakeProvider("primary", fn)
        gw = _gateway([p])
        gw.chat([{"role": "user", "content": "hi"}])
        assert state["n"] == 2

    def test_backoff_timing(self):
        sleeps = []
        state = {"n": 0}
        def fn(messages, temperature):
            state["n"] += 1
            if state["n"] < 3:
                raise RetryableProviderError("500")
            return _ok()
        p = FakeProvider("primary", fn)
        gw = _gateway([p], retry_policy=_retry(sleep_rec=sleeps.append))
        gw.chat([{"role": "user", "content": "hi"}])
        assert sleeps == [1.0, 2.0]  # base=1, mult=2, jitter off


# ── Circuit breaker ──


class TestCircuitBreaker:
    def _gateway(self, threshold=2, cooldown=5.0, clock=None, max_retries=0):
        provs = [FakeProvider("primary", lambda m, t: (_ for _ in ()).throw(RetryableProviderError("down")))]
        return _gateway(provs,
                        retry_policy=_retry(max_retries=max_retries),
                        circuit_config=CircuitConfig(failure_threshold=threshold,
                                                     cooldown_seconds=cooldown),
                        now=clock)

    def test_circuit_opens_after_n_failures(self):
        gw = self._gateway(threshold=2)
        p = gw.providers[0]
        gw.chat([]); gw.chat([])                       # 2 failures → OPEN
        assert gw._breaker("primary").state == CircuitBreaker.OPEN
        before = p.calls
        text, raw = gw.chat([])                        # OPEN → skip, canned fallback
        assert p.calls == before                        # provider not invoked
        assert raw["gateway_fallback"] is True

    def test_half_open_probe_recovers(self):
        clock = FakeClock()
        gw = self._gateway(threshold=2, cooldown=5.0, clock=clock)
        gw.chat([]); gw.chat([])                       # OPEN
        clock.advance(6.0)                             # cooldown passed
        gw.providers[0]._call = lambda m, t: _ok("recovered")
        text, raw = gw.chat([])                        # probe succeeds
        assert text == "recovered"
        assert gw._breaker("primary").state == CircuitBreaker.CLOSED

    def test_half_open_probe_failure_reopens(self):
        clock = FakeClock()
        gw = self._gateway(threshold=2, cooldown=5.0, clock=clock)
        gw.chat([]); gw.chat([])                       # OPEN
        clock.advance(6.0)
        gw.chat([])                                    # probe fails → reopen
        assert gw._breaker("primary").state == CircuitBreaker.OPEN


# ── Failover & fallback ──


class TestFailover:
    def test_fallback_to_provider2(self):
        def bad(messages, temperature):
            raise RetryableProviderError("down")
        def good(messages, temperature):
            return _ok("from_backup", "m2")
        p1 = FakeProvider("primary", bad)
        p2 = FakeProvider("backup_2", good)
        gw = _gateway([p1, p2], retry_policy=_retry(max_retries=1))
        text, raw = gw.chat([])
        assert text == "from_backup"
        assert raw["provider"] == "backup_2"

    def test_all_providers_down_graceful(self):
        def bad(messages, temperature):
            raise RetryableProviderError("down")
        gw = _gateway([FakeProvider("a", bad), FakeProvider("b", bad)],
                      retry_policy=_retry(max_retries=0))
        text, raw = gw.chat([])
        assert raw["gateway_fallback"] is True
        assert len(raw["errors"]) == 2
        assert "无法回答此问题" in text

    def test_raise_mode(self):
        def bad(messages, temperature):
            raise RetryableProviderError("down")
        gw = _gateway([FakeProvider("a", bad)], retry_policy=_retry(max_retries=0),
                      raise_on_total_failure=True)
        with pytest.raises(LLMUnavailableError):
            gw.chat([])


# ── Error classification ──


class TestClassification:
    def test_classify_real_openai_exceptions(self):
        import httpx
        import openai
        req = httpx.Request("POST", "http://example.com/v1/chat/completions")
        def resp(status):
            return httpx.Response(status_code=status, request=req)

        assert _classify_exc(openai.RateLimitError("rl", response=resp(429), body=None)) is True
        assert _classify_exc(openai.InternalServerError("is", response=resp(500), body=None)) is True
        assert _classify_exc(openai.APITimeoutError(request=req)) is True
        assert _classify_exc(openai.APIConnectionError(message="c", request=req)) is True
        assert _classify_exc(openai.BadRequestError("br", response=resp(400), body=None)) is False

    def test_is_retryable_status(self):
        assert _is_retryable_status(429) is True
        assert _is_retryable_status(500) is True
        assert _is_retryable_status(503) is True
        assert _is_retryable_status(400) is False
        assert _is_retryable_status(401) is False

    def test_timeout_path_retries(self):
        state = {"n": 0}
        def fn(messages, temperature):
            state["n"] += 1
            if state["n"] == 1:
                raise RetryableProviderError("APITimeoutError")
            return _ok()
        gw = _gateway([FakeProvider("primary", fn)])
        text, raw = gw.chat([])
        assert text == "ok"
        assert state["n"] == 2


# ── LLMClient facade ──


class TestFacade:
    def test_llm_client_delegates_to_gateway(self):
        captured = {}
        class StubGw:
            def chat(self, messages, temperature=0.0):
                captured["messages"] = messages
                captured["temperature"] = temperature
                return "stub", {"model": "m"}
        with patch("src.infra.gateway.get_gateway", return_value=StubGw()):
            from src.infra.llm_client import LLMClient
            text, raw = LLMClient().chat([{"role": "user", "content": "hi"}], temperature=0.3)
        assert text == "stub"
        assert captured["temperature"] == 0.3
        assert captured["messages"][0]["content"] == "hi"

    def test_llm_client_explicit_args_isolated(self):
        fake = FakeProvider("custom", lambda m, t: ("iso", {"model": "custom"}))
        with patch("src.infra.gateway.Provider", return_value=fake), \
             patch("src.infra.gateway.get_gateway") as mock_shared:
            from src.infra.llm_client import LLMClient
            text, raw = LLMClient("http://custom", "k", "m").chat([{"role": "user", "content": "x"}])
        assert text == "iso"
        mock_shared.assert_not_called()


# ── Provider chain from settings ──


class TestBuildProviders:
    def test_build_providers_from_settings(self, monkeypatch):
        from src.config import settings as s
        monkeypatch.setattr(s, "llm_base_url", "http://p1/v1")
        monkeypatch.setattr(s, "llm_api_key", "k1")
        monkeypatch.setattr(s, "llm_model", "m1")
        monkeypatch.setattr(s, "llm_base_url_2", "http://p2/v1")
        monkeypatch.setattr(s, "llm_api_key_2", "k2")
        monkeypatch.setattr(s, "llm_model_2", "m2")
        monkeypatch.setattr(s, "llm_base_url_3", "https://api.example.com/v1")
        monkeypatch.setattr(s, "llm_api_key_3", "replace-me")
        monkeypatch.setattr(s, "llm_model_3", "replace-me")
        provs = _build_providers(s)
        assert [p.cfg.name for p in provs] == ["primary", "backup_2"]


# ── Thread safety ──


class TestThreadSafety:
    def test_thread_safety_smoke(self):
        gw = _gateway([FakeProvider("primary", lambda m, t: _ok())])
        results = []
        errors = []

        def worker():
            try:
                text, raw = gw.chat([])
                results.append(raw["provider"])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(results) == 20
        assert gw._breaker("primary").state == CircuitBreaker.CLOSED

    def test_thread_safety_single_probe(self):
        clock = FakeClock()
        gw = _gateway([FakeProvider("primary", lambda m, t: (_ for _ in ()).throw(RetryableProviderError("down")))],
                      retry_policy=_retry(max_retries=0),
                      circuit_config=CircuitConfig(failure_threshold=1, cooldown_seconds=5))
        gw.chat([])                      # failure 1 → OPEN
        clock.advance(6.0)               # cooldown passed
        p = gw.providers[0]
        threads = [threading.Thread(target=gw.chat, args=([],)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert p.calls == 1              # exactly ONE HALF_OPEN probe


# ── Outage message short-circuits VerifiedQA ──


class TestVerifiedQAIntegration:
    def test_canned_answer_shortcircuits_verifiedqa(self):
        """The gateway fallback contains '无法回答此问题' → verify skips the
        verifier and the pipeline refuses instead of fabricating."""
        from src.workflow.verified_qa import VerifiedQA
        chunks = [{"chunk_id": "c1", "page_number": 24, "content_type": "text",
                   "content": "电池电量不足请充电。", "source_file": "m.pdf",
                   "rerank_score": 0.5}]
        retriever = MagicMock()
        retriever.search.return_value = chunks
        verifier = MagicMock()

        def generator(question, cs):
            return {"answer": "模型服务暂时不可用，无法回答此问题，请稍后重试。"}

        vqa = VerifiedQA(retriever, generator, verifier, max_retries=1)
        state = vqa.run("问题")
        assert state["final_status"] == "refused"
        verifier.assert_not_called()      # verifier never invoked during outage
