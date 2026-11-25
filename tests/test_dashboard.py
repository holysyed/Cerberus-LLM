"""Operator dashboard behavior on real, legacy and unavailable audit files."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from cerberus.core.audit import AuditStore
from examples.demo import run

SCRIPT = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_activity_correlates_attempts_and_persists_only_known_types(tmp_path: Path) -> None:
    """A later accepted candidate supersedes an earlier block under the same ID."""
    path = tmp_path / "audit.sqlite3"
    url = f"sqlite:///{path}"
    with closing(AuditStore(url)) as store:
        store.record("retry", "schema", "schema", 0.9, False, 1)
        store.record("retry", "pipeline", "blocked", 0.9, False, 2)
        store.record("other", "input", "prompt_injection", 0.92, False, 1)
        store.record("other", "pipeline", "blocked", 0.92, False, 2)
        store.record(
            "retry",
            "pii",
            "pii",
            0.85,
            False,
            1,
            pii_types=("email", "name", "private@example.invalid"),
        )
        store.record("retry", "pipeline", "allowed", 0.85, True, 3)
        store.record("retry", "pii", "pii", 0.85, False, 1, pii_types=("secret",))
    with closing(AuditStore(url, read_only=True)) as reader:
        latest, other = reader.request_activity()
        assert latest.request_id == "retry" and not latest.blocked
        assert latest.guard_ms == 5 and latest.risk == 0.9
        assert latest.pii_types == {"email", "name"} and latest.stages == {"Output"}
        assert other.blocked and other.block_category == "prompt_injection"
        assert reader.request_activity(1) == [latest]
        with pytest.raises(ValueError):
            reader.request_activity(0)
        with pytest.raises(OperationalError):
            reader.save_report("unwritable", {})
    assert b"private@example.invalid" not in path.read_bytes()


def test_reader_opens_legacy_audits_without_mutating_them(tmp_path: Path) -> None:
    """Historical guard rows remain visible with unknown PII types omitted."""
    path = tmp_path / "legacy.sqlite3"
    url = f"sqlite:///{path}"
    with closing(AuditStore(url)) as store:
        store.record("legacy", "pii", "pii", 0.85, False, 1)
        store.record("legacy", "pipeline", "allowed", 0.85, True, 1)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP TABLE pii_detection_types")
        conn.commit()
    with closing(AuditStore(url, read_only=True)) as reader:
        (row,) = reader.request_activity()
        assert row.request_id == "legacy" and not row.pii_types
    with closing(AuditStore(url)) as upgraded:
        assert len(upgraded.request_activity()) == 1


def test_dashboard_views_filter_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real Streamlit script and its two operator views."""
    testing = pytest.importorskip("streamlit.testing.v1")
    path = tmp_path / "demo.sqlite3"
    assert run(path)[:3] == (12, 3, 3)
    monkeypatch.setenv("CERBERUS_DATABASE_URL", f"sqlite:///{path}")
    app = testing.AppTest.from_file(str(SCRIPT), default_timeout=20).run()
    assert not app.exception and not app.warning
    app.radio(key="activity-outcome").set_value("Blocked").run()
    assert not app.exception
    app.radio[0].set_value("Reports").run()
    assert not app.exception
    assert len(app.dataframe) == 2 and len(app.get("download_button")) == 3
    assert len(app.dataframe[0].value) == 3


def test_dashboard_empty_and_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty and uninitialized stores give actionable state without creating demo data."""
    testing = pytest.importorskip("streamlit.testing.v1")
    path = tmp_path / "empty.sqlite3"
    url = f"sqlite:///{path}"
    monkeypatch.setenv("CERBERUS_DATABASE_URL", url)
    app = testing.AppTest.from_file(str(SCRIPT), default_timeout=20).run()
    assert app.warning and not path.exists()
    AuditStore(url).close()
    app.run()
    assert app.info and not app.exception
    app.radio[0].set_value("Reports").run()
    assert app.info and not app.exception
