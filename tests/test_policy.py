"""Configuration boundary and packaged profile tests."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cerberus.core.policy import GuardPolicy, Policy, load_policy


def test_bundled_policy_and_fallback() -> None:
    """Bundled defaults enable new guards and fail closed."""
    policy = load_policy()
    assert policy.mode == "sanitize"
    assert policy.fail_mode == "closed"
    assert policy.for_guard("input") == GuardPolicy()


def test_override_and_filesystem_loading(tmp_path: Path) -> None:
    """Per-guard options survive a file load, including a zero threshold."""
    source = tmp_path / "local.yaml"
    source.write_text(
        "mode: sanitize\nfail_mode: open\nguards:\n  pii:\n    enabled: false\n    threshold: 0\n",
        encoding="utf-8",
    )
    policy = load_policy(source)
    assert (policy.mode, policy.fail_mode) == ("sanitize", "open")
    assert policy.for_guard("pii") == GuardPolicy(enabled=False, threshold=0.0)


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "drop"},
        {"fail_mode": "ignore"},
        {"disabled": True},
        {"threshold": "0.5"},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": float("nan")},
        {"threshold": 0.9, "critical_threshold": 0.5},
        {"guards": {"pii": {"threshold": 1.0}}},
        {"guards": {"pii": {"enable": True}}},
        {"guards": {"pii": {"enabled": "true"}}},
        {"guards": {"": {}}},
    ],
)
def test_invalid_policy(value: dict[str, object]) -> None:
    """Reject ambiguous, misspelled, or out-of-range settings."""
    with pytest.raises(ValidationError):
        Policy.model_validate(value)


@pytest.mark.parametrize(
    "content",
    [
        "!!python/tuple []",
        "guards:\n  pii:\n    enabled: false\n    enabled: true\n",
        "mode: warn\n---\nmode: block",
    ],
)
def test_reject_unsafe_or_ambiguous_yaml(tmp_path: Path, content: str) -> None:
    """Disallow object tags, duplicate keys, and trailing documents."""
    source = tmp_path / "invalid.yaml"
    source.write_text(content, encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_policy(str(source))
