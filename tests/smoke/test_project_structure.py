from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_governance_documents_exist() -> None:
    required = [
        "PROJECT_CHARTER.md",
        "ARCHITECTURE.md",
        "DATA_CONTRACTS.md",
        "ROADMAP.md",
        "ACCEPTANCE_MATRIX.md",
        "EXPERIMENT_PROTOCOL.md",
        "EXPERIMENT_LOG.md",
        "DECISIONS.md",
        "PROGRESS.md",
        "SESSION_LOG.md",
    ]
    assert all((ROOT / "docs" / name).exists() for name in required)


def test_experiment_profiles_exist() -> None:
    profiles = [
        "v0_naive.yaml",
        "v1_multimodal.yaml",
        "v2_hybrid.yaml",
        "v3_rerank.yaml",
        "v4_verified.yaml",
    ]
    assert all((ROOT / "configs" / "experiments" / name).exists() for name in profiles)
