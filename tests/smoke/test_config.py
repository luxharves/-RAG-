"""Smoke tests for configuration loading."""

import os
from pathlib import Path


def test_settings_import() -> None:
    """Settings module should import without error."""
    from src.config.settings import settings

    assert settings is not None


def test_settings_defaults() -> None:
    """Settings should load and provide expected attributes."""
    from src.config.settings import settings

    # Attributes must exist (values may differ from class defaults due to .env)
    assert hasattr(settings, "embedding_model")
    assert hasattr(settings, "reranker_model")
    assert hasattr(settings, "model_device")
    assert hasattr(settings, "log_level")
    # Non-empty values
    assert len(settings.embedding_model) > 0
    assert len(settings.reranker_model) > 0


def test_settings_project_root() -> None:
    """project_root should point to the actual project directory."""
    from src.config.settings import settings

    root = settings.project_root
    assert root.exists()
    assert (root / "docs").exists()
    assert (root / "configs").exists()


def test_settings_env_override(monkeypatch) -> None:
    """Environment variables should override defaults."""
    import sys
    from importlib import reload

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("EMBEDDING_MODEL", "test/model")

    reload(sys.modules["src.config.settings"])
    from src.config.settings import settings

    assert settings.log_level == "DEBUG"
    assert settings.embedding_model == "test/model"

    # Restore original
    monkeypatch.undo()
    reload(sys.modules["src.config.settings"])


def test_sensitive_keys_not_in_defaults() -> None:
    """The Settings *class defaults* must not contain real secrets."""
    from src.config.settings import Settings

    # Check class-level field defaults, not .env-overridden instance values
    default_llm_key = Settings.model_fields["llm_api_key"].default
    default_vlm_key = Settings.model_fields["vlm_api_key"].default

    assert default_llm_key == "replace-me"
    assert default_vlm_key == "replace-me"
    assert not default_llm_key.startswith("sk-")
