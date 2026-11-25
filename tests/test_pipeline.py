"""Exercise the public guard pipeline contract across enforcement modes."""

from uuid import UUID

import pytest
from pydantic import ValidationError

from cerberus.core.audit import AuditStore
from cerberus.core.models import Guard, GuardContext, GuardResult
from cerberus.core.pipeline import Pipeline
from cerberus.core.policy import GuardPolicy, Policy


class Probe(Guard):
    """A deterministic guard that records what it received."""

    def __init__(self, name: str, risk: float = 0.0, replacement: str | None = None) -> None:
        """Configure this probe's score and optional safe replacement."""
        self.name = name
        self.risk = risk
        self.replacement = replacement
        self.seen: list[str] = []

    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Observe the input and return a predictable decision."""
        self.seen.append(text)
        return GuardResult(
            passed=self.risk < 0.65,
            risk_score=self.risk,
            category="test",
            sanitized_text=self.replacement,
        )


class Broken(Guard):
    """An unavailable backend with a sensitive error message."""

    name = "broken"

    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Simulate an SDK exception that embeds caller data."""
        raise RuntimeError(text)


def test_block_short_circuits() -> None:
    """A blocked candidate is never passed on or returned."""
    second = Probe("second")
    decision = Pipeline([Probe("first", 0.65), second], Policy()).run("sample")
    assert decision.blocked and not decision.text
    assert not second.seen
    assert len(decision.results) == 1
    assert decision.results[0].latency_ms >= 0


def test_warn_continues_until_critical() -> None:
    """Warnings retain original content and a critical decision always stops."""
    second = Probe("second", 0.95)
    third = Probe("third")
    decision = Pipeline([Probe("first", 0.7, "new"), second, third], Policy(mode="warn")).run("old")
    assert second.seen == ["old"]
    assert decision.blocked and not third.seen
    assert all(not result.passed for result in decision.results)


@pytest.mark.parametrize("replacement", ["", "[REDACTED]"])
def test_sanitize_forwards_only_replacement(replacement: str) -> None:
    """Sanitized content, including a legitimate empty string, reaches later guards."""
    observer = Probe("observer")
    pipeline = Pipeline([Probe("pii", 0.8, replacement), observer], Policy(mode="sanitize"))
    decision = pipeline.run("private")
    assert not decision.blocked and decision.text == replacement
    assert observer.seen == [replacement]
    assert all(result.sanitized_text is None for result in decision.results)


def test_later_block_suppresses_earlier_replacement() -> None:
    """A partially sanitized intermediate value cannot surface after a later block."""
    pipeline = Pipeline(
        [Probe("first", 0.7, "private remainder"), Probe("second", 0.7)], Policy(mode="sanitize")
    )
    decision = pipeline.run("private original")
    assert decision.blocked and "private" not in decision.model_dump_json()


def test_disabled_guards_and_policy_threshold() -> None:
    """Per-guard configuration governs both invocation and threshold boundaries."""
    disabled = Probe("off", 1.0)
    policy = Policy(guards={"off": GuardPolicy(enabled=False), "other": GuardPolicy(threshold=0.8)})
    decision = Pipeline([disabled, Probe("other", 0.7)], policy).run("benign")
    assert not disabled.seen
    assert not decision.blocked and decision.results[0].passed
    assert decision.text == "benign"


@pytest.mark.parametrize("fail_closed", [False, True])
def test_guard_unavailability(fail_closed: bool, caplog: pytest.LogCaptureFixture) -> None:
    """Backend exceptions follow availability policy without disclosing the error."""
    observer = Probe("second")
    policy = Policy(fail_mode="closed" if fail_closed else "open", mode="warn")
    decision = Pipeline([Broken(), observer], policy).run("private test value")
    assert decision.blocked is fail_closed
    assert decision.results[0].category == "availability"
    assert decision.results[0].passed is not fail_closed
    assert bool(observer.seen) is not fail_closed
    assert "private test value" not in caplog.text


def test_reject_duplicate_guard_names() -> None:
    """A policy override must refer to a unique guard in a pipeline."""
    with pytest.raises(ValueError, match="unique"):
        Pipeline([Probe("a"), Probe("a")], Policy())


def test_context_and_results_validation() -> None:
    """Correlation IDs, finite scores, and valid timing are part of the contract."""
    context = GuardContext()
    assert UUID(context.request_id).version == 4
    assert Pipeline([], Policy()).run("text", context).request_id == context.request_id
    with pytest.raises(ValidationError):
        GuardContext(request_id="plain text")
    with pytest.raises(ValidationError):
        GuardResult(passed=True, risk_score=float("nan"), category="test")
    with pytest.raises(ValidationError):
        GuardResult(passed=True, risk_score=0, category="test", latency_ms=-1)


def test_sqlite_records_shared_request_id() -> None:
    """Each configured stage and final decision share one correlation ID."""
    store = AuditStore("sqlite://")
    try:
        decision = Pipeline([Probe("first")], Policy(), store).run("sample")
        rows = store.recent()
        assert [row["guard"] for row in rows] == ["pipeline", "first"]
        assert {row["request_id"] for row in rows} == {decision.request_id}
    finally:
        store.close()
