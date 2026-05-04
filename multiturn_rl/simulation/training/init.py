from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    return project_root() / "simulation" / "artifacts"


def phase10_artifacts_root() -> Path:
    out = artifacts_root() / "phase10"
    out.mkdir(parents=True, exist_ok=True)
    return out
