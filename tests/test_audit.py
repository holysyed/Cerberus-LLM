"""Tests for privacy-preserving logging, persistence and concurrent writes."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from cerberus.core.audit import AuditStore, log_decision


def test_json_log(caplog: pytest.LogCaptureFixture) -> None:
    """All fields are metadata and accidental free-form labels are removed."""
    with caplog.at_level(logging.INFO, logger="cerberus.audit"):
        log_decision("request-1", "input", "accidental user@example.test", 0.7, False, 0.8)
    fields = json.loads(caplog.records[-1].message)
    assert fields["event"] == "guard_decision"
    assert fields["request_id"] == "request-1"
    assert fields["category"] == "unclassified"
    assert "example.test" not in caplog.text


def test_persistence_and_report_atomicity(tmp_path: Path) -> None:
    """Decisions survive reopening and duplicate report IDs preserve the original."""
    url = f"sqlite:///{(tmp_path / 'audit.sqlite').as_posix()}"
    store = AuditStore(url)
    store.record("a", "input", "benign", 0, True, 1)
    store.record("b", "output", "pii", 0.95, False, 2)
    store.save_report("run-1", {"cases": 3})
    with pytest.raises(IntegrityError):
        store.save_report("run-1", {"cases": 5})
    store.close()
    reopened = AuditStore(url)
    assert [row["request_id"] for row in reopened.recent()] == ["b", "a"]
    assert reopened.recent(1)[0]["passed"] is False
    assert reopened.get_report("run-1") == {"cases": 3}
    assert reopened.get_report("missing") is None
    assert reopened.recent_reports()[0]["id"] == "run-1"
    reopened.close()


def test_store_boundaries() -> None:
    """Reject unsupported databases, oversized queries and invalid report IDs."""
    with pytest.raises(ValueError):
        AuditStore("postgresql://localhost/example")
    store = AuditStore("sqlite://")
    with pytest.raises(ValueError):
        store.recent(0)
    with pytest.raises(ValueError):
        store.recent_reports(1001)
    with pytest.raises(ValueError):
        store.save_report("invalid label", {})
    store.close()


def test_shared_memory_store() -> None:
    """Worker threads can share an isolated in-memory database."""
    store = AuditStore("sqlite:///:memory:")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: store.record(str(i), "input", "ok", 0, True, 1), range(16)))
    assert len({row["request_id"] for row in store.recent()}) == 16
    store.close()
