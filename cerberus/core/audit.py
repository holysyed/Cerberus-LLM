"""Structured decision logging and SQLite storage for audit metadata."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    inspect,
    literal,
    select,
)
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool

_logger = logging.getLogger("cerberus.audit")
_identifier = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,95}")
_metadata = MetaData()
_decisions = Table(
    "guard_decisions",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("timestamp", String(40), nullable=False),
    Column("request_id", String(96), nullable=False, index=True),
    Column("guard", String(96), nullable=False),
    Column("category", String(96), nullable=False),
    Column("risk_score", Float, nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("latency_ms", Float, nullable=False),
)
_reports = Table(
    "evaluation_reports",
    _metadata,
    Column("id", String(96), primary_key=True),
    Column("timestamp", String(40), nullable=False),
    Column("payload", JSON, nullable=False),
)

_pii_types = Table(
    "pii_detection_types",
    _metadata,
    Column("decision_id", Integer, primary_key=True),
    Column("kind", String(12), primary_key=True),
)
_KNOWN_PII_TYPES = frozenset(
    {"aadhaar", "card", "email", "ifsc", "name", "pan", "phone", "secret", "ssn"}
)


@dataclass
class RequestActivity:
    """Metadata from completed pipelines sharing one request ID."""

    request_id: str
    timestamp: str = ""
    blocked: bool = False
    block_category: str = ""
    risk: float = 0.0
    guard_ms: float = 0.0
    pii_types: set[str] = field(default_factory=set)
    stages: set[str] = field(default_factory=set)


def _safe_id(value: str) -> str:
    """Replace accidental free-form event labels with a fixed identifier."""
    return value if _identifier.fullmatch(value) else "unclassified"


def _event(
    request_id: str, guard: str, category: str, risk_score: float, passed: bool, latency_ms: float
) -> dict[str, object]:
    """Build the fields common to the log event and database record."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": _safe_id(request_id),
        "guard": _safe_id(guard),
        "category": _safe_id(category),
        "risk_score": risk_score,
        "passed": passed,
        "latency_ms": latency_ms,
    }


def log_decision(
    request_id: str, guard: str, category: str, risk_score: float, passed: bool, latency_ms: float
) -> None:
    """Emit one decision as a JSON object through the cerberus.audit logger."""
    fields = _event(request_id, guard, category, risk_score, passed, latency_ms)
    _logger.info(json.dumps({"event": "guard_decision", **fields}, allow_nan=False))


class AuditStore:
    """Synchronous transaction-scoped SQLite storage shared by the gateway and dashboard."""

    def __init__(self, database_url: str, *, read_only: bool = False) -> None:
        """Open a SQLite database and initialize its tables."""
        url = make_url(database_url)
        if url.get_backend_name() != "sqlite":
            raise ValueError("AuditStore requires a SQLite URL")
        if read_only:
            if not url.database or url.database == ":memory:" or url.query:
                raise ValueError("The dashboard needs a file-backed SQLite database")
            uri = Path(url.database).resolve().as_uri() + "?mode=ro"
            self._engine = create_engine(
                "sqlite://",
                creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5),
            )
        else:
            self._engine = create_engine(
                url,
                connect_args={"check_same_thread": False, "timeout": 10},
                poolclass=StaticPool if url.database in (None, "", ":memory:") else None,
            )
        # An in-memory store has one connection shared by all worker threads.
        self._lock = RLock()
        if not read_only:
            _metadata.create_all(self._engine)
        self._has_types = not read_only or inspect(self._engine).has_table(_pii_types.name)

    def record(
        self,
        request_id: str,
        guard: str,
        category: str,
        risk_score: float,
        passed: bool,
        latency_ms: float,
        *,
        pii_types: Sequence[str] = (),
    ) -> None:
        """Persist a decision without accepting prompt text or guard reasons."""
        fields = _event(request_id, guard, category, risk_score, passed, latency_ms)
        with self._lock, self._engine.begin() as conn:
            event = conn.execute(_decisions.insert(), fields)
            kinds = _KNOWN_PII_TYPES.intersection(pii_types) if guard == "pii" else set()
            if kinds:
                conn.execute(
                    _pii_types.insert(),
                    [{"decision_id": event.lastrowid, "kind": kind} for kind in sorted(kinds)],
                )

    def request_activity(self, limit: int = 100) -> list[RequestActivity]:
        """Summarize the most recent distinct requests up to their latest completed stage."""
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        d = _decisions.c
        last = func.max(d.id).label("last_id")
        recent = (
            select(d.request_id, last)
            .where(d.guard == "pipeline")
            .group_by(d.request_id)
            .order_by(last.desc())
            .limit(limit)
            .cte("recent")
        )
        source = _decisions.join(
            recent, (d.request_id == recent.c.request_id) & (d.id <= recent.c.last_id)
        )
        if self._has_types:
            source = source.outerjoin(_pii_types, d.id == _pii_types.c.decision_id)
        kind = _pii_types.c.kind if self._has_types else literal(None)
        stmt = (
            select(_decisions, kind.label("kind"))
            .select_from(source)
            .order_by(recent.c.last_id.desc(), d.id)
        )
        groups: dict[str, RequestActivity] = {}
        previous_guard = ""
        with self._lock, self._engine.connect() as conn:
            for row in conn.execute(stmt).mappings():
                key = row["request_id"]
                if key not in groups:
                    groups[key] = RequestActivity(key)
                    previous_guard = ""
                request = groups[key]
                if row["guard"] == "pipeline":
                    request.timestamp = row["timestamp"]
                    request.blocked = not row["passed"]
                    request.block_category = previous_guard if request.blocked else ""
                    request.guard_ms += row["latency_ms"]
                    request.risk = max(request.risk, row["risk_score"])
                    previous_guard = ""
                else:
                    previous_guard = row["category"]
                    request.stages.add("Input" if row["guard"] == "input" else "Output")
                    if row["kind"]:
                        request.pii_types.add(row["kind"])
        return list(groups.values())

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the newest decisions first, up to 1,000 per query."""
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = select(_decisions).order_by(_decisions.c.id.desc()).limit(limit)
        with self._lock, self._engine.connect() as conn:
            return [dict(row) for row in conn.execute(query).mappings()]

    def save_report(self, report_id: str, payload: dict[str, Any]) -> None:
        """Insert an evaluator-produced JSON report; duplicate IDs cannot replace reports."""
        if not _identifier.fullmatch(report_id):
            raise ValueError("Invalid report identifier")
        fields = {"id": report_id, "timestamp": datetime.now(UTC).isoformat(), "payload": payload}
        with self._lock, self._engine.begin() as conn:
            conn.execute(_reports.insert(), fields)

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Retrieve a stored report by ID."""
        query = select(_reports.c.payload).where(_reports.c.id == report_id)
        with self._lock, self._engine.connect() as conn:
            payload = conn.execute(query).scalar_one_or_none()
            return None if payload is None else dict(payload)

    def recent_reports(self, limit: int = 100) -> list[dict[str, Any]]:
        """List report identifiers and creation times, newest first."""
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = (
            select(_reports.c.id, _reports.c.timestamp)
            .order_by(_reports.c.timestamp.desc(), _reports.c.id.desc())
            .limit(limit)
        )
        with self._lock, self._engine.connect() as conn:
            return [dict(row) for row in conn.execute(query).mappings()]

    def close(self) -> None:
        """Release database connections held by this store."""
        with self._lock:
            self._engine.dispose()
