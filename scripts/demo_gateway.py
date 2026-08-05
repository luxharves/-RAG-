#!/usr/bin/env python
"""V7 — LLM gateway failover demo.

Builds its OWN gateway (never touches the app's get_gateway singleton):
  - dead_primary: a closed local port → connection refused → retries → failover
  - backup: the real configured LLM from settings (DeepSeek)

Flow: queries 1-2 retry the dead primary then answer via backup; query 3+ the
primary's circuit is OPEN (skipped instantly); after cooldown a HALF_OPEN probe
fires, fails, and re-opens. Ends with a per-provider state dump.

Usage:
    python scripts/demo_gateway.py [--queries 6]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings  # noqa: E402
from src.infra.gateway import (  # noqa: E402
    CircuitConfig, LLMGateway, Provider, ProviderConfig, RetryPolicy,
)

DEAD = ProviderConfig(
    "dead_primary", "http://127.0.0.1:59999/v1", "sk-fake", "deepseek-chat",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 gateway failover demo")
    parser.add_argument("--queries", type=int, default=6)
    args = parser.parse_args()

    backup = ProviderConfig(
        "backup", settings.llm_base_url, settings.llm_api_key, settings.llm_model,
    )
    if not backup.is_configured():
        print("⚠️  backup provider 未配置 (settings 是占位符) — 演示会走 all-providers-down 兜底路径。")
        gateway = LLMGateway([Provider(DEAD, timeout=3.0)], retry_policy=_policy(),
                             circuit_config=CircuitConfig(failure_threshold=2, cooldown_seconds=5))
    else:
        gateway = LLMGateway([Provider(DEAD, timeout=3.0), Provider(backup, timeout=60.0)],
                             retry_policy=_policy(),
                             circuit_config=CircuitConfig(failure_threshold=2, cooldown_seconds=5))

    print(f"网关演示: dead_primary(死端口) → backup({backup.name}: {backup.model})\n")

    for i in range(1, args.queries + 1):
        t0 = time.perf_counter()
        text, raw = gateway.chat([{"role": "user", "content": "说一个字"}])
        elapsed = time.perf_counter() - t0
        provider = raw.get("provider", "none")
        if raw.get("gateway_fallback"):
            print(f"[{i}/{args.queries}] 兜底应答 ({elapsed:.2f}s) — {text[:30]}")
        else:
            print(f"[{i}/{args.queries}] provider={provider} attempts={raw.get('attempts')} "
                  f"({elapsed:.2f}s) — {text[:30]}")
        if i == args.queries // 2:
            print(f"  … 等待 {CircuitConfig(failure_threshold=2, cooldown_seconds=5).cooldown_seconds}s "
                  f"冷却,触发 HALF_OPEN 探针 …")
            time.sleep(CircuitConfig(failure_threshold=2, cooldown_seconds=5).cooldown_seconds + 1)

    print("\n── 熔断状态 ──")
    for p in gateway.state_dump()["providers"]:
        print(f"  {p['name']:<14} state={p['state']:<10} failures={p['consecutive_failures']} "
              f"seconds_until_half_open={p['seconds_until_half_open']}")


def _policy() -> RetryPolicy:
    return RetryPolicy(max_retries=2, base=0.3, multiplier=2, cap=1.5, jitter=False)


if __name__ == "__main__":
    main()
