"""Ordered guard execution with policy enforcement and correlated audit events."""

from collections.abc import Sequence
from time import perf_counter

from cerberus.core.audit import AuditStore, log_decision
from cerberus.core.models import Guard, GuardContext, GuardResult, PipelineResult
from cerberus.core.policy import Policy


class Pipeline:
    """Apply a policy to enabled guards and release only the final text."""

    def __init__(
        self, guards: Sequence[Guard], policy: Policy, audit_store: AuditStore | None = None
    ) -> None:
        """Capture an ordered set of uniquely named guards."""
        self.guards = tuple(guards)
        names = [guard.name for guard in self.guards]
        if len(set(names)) != len(names):
            raise ValueError("Guard names must be unique within a pipeline")
        self.policy = policy
        self.audit_store = audit_store

    def _record(
        self,
        request_id: str,
        guard: str,
        category: str,
        risk: float,
        passed: bool,
        elapsed: float,
        pii_types: tuple[str, ...] = (),
    ) -> None:
        """Write decision metadata to the logger and configured audit store."""
        log_decision(request_id, guard, category, risk, passed, elapsed)
        if self.audit_store is not None:
            self.audit_store.record(
                request_id, guard, category, risk, passed, elapsed, pii_types=pii_types
            )

    def run(self, text: str, context: GuardContext | None = None) -> PipelineResult:
        """Score each guard against policy, short-circuiting any blocking decision."""
        started = perf_counter()
        context = context or GuardContext()
        decisions: list[GuardResult] = []
        blocked = False

        for guard in self.guards:
            config = self.policy.for_guard(guard.name)
            if not config.enabled:
                continue
            check_started = perf_counter()
            try:
                result = guard.check(text, context)
            except Exception:  # noqa: BLE001 - isolate third-party guard failures
                # Backend errors may contain raw inputs, API keys, or provider bodies.
                blocked = self.policy.fail_mode == "closed"
                result = GuardResult(
                    passed=not blocked,
                    risk_score=1.0 if blocked else 0.0,
                    category="availability",
                    reasons=("guard_unavailable",),
                )
            else:
                threshold = (
                    config.threshold if config.threshold is not None else self.policy.threshold
                )
                passed = result.risk_score < threshold
                result = result.model_copy(update={"passed": passed})
                if result.risk_score >= self.policy.critical_threshold:
                    blocked = True
                elif not passed:
                    if self.policy.mode == "sanitize" and result.sanitized_text is not None:
                        text = result.sanitized_text
                    else:
                        blocked = self.policy.mode != "warn"

            # A later guard may block material an earlier one included in its replacement.
            result = result.model_copy(
                update={
                    "latency_ms": (perf_counter() - check_started) * 1000,
                    "sanitized_text": None,
                }
            )
            decisions.append(result)
            self._record(
                context.request_id,
                guard.name,
                result.category,
                result.risk_score,
                result.passed,
                result.latency_ms,
                result.reasons if guard.name == "pii" else (),
            )
            if blocked:
                break

        elapsed = (perf_counter() - started) * 1000
        self._record(
            context.request_id,
            "pipeline",
            "blocked" if blocked else "allowed",
            max((decision.risk_score for decision in decisions), default=0.0),
            not blocked,
            elapsed,
        )
        return PipelineResult(
            request_id=context.request_id,
            blocked=blocked,
            text="" if blocked else text,
            results=tuple(decisions),
            latency_ms=elapsed,
        )
