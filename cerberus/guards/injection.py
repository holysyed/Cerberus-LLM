"""Layered injection detection across independent application trust boundaries."""

from collections.abc import Callable
from time import perf_counter

from cerberus.core.models import Guard, GuardContext, GuardResult
from cerberus.core.policy import Policy
from cerberus.guards.detectors import HFClassifier
from cerberus.guards.heuristics import heuristic_score

Detector = Callable[[str], float]


class InputGuard(Guard):
    """Combine evidence per source, retaining the highest source risk."""

    name = "input"

    def __init__(
        self, policy: Policy, classifier: Detector | None = None, judge: Detector | None = None
    ) -> None:
        """Configure optional detectors once, leaving model loading to first inference."""
        self.policy = policy
        config = policy.input
        self.layers: list[tuple[str, Detector | None, float, float]] = []
        if config.ml_enabled:
            backend = HFClassifier() if classifier is None else classifier
            self.layers.append(("ml", backend, config.ml_threshold, config.ml_weight))
        if config.judge_enabled:
            self.layers.append(("judge", judge, config.judge_threshold, config.judge_weight))

    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Evaluate the message, each retrieved document, and each tool result."""
        started = perf_counter()
        config = self.policy.input
        source_count = 1 + len(context.documents) + len(context.tool_outputs)
        if source_count > config.max_sources:
            return self._result(1.0, "input_limit", ["source_count_exceeded"], started)

        sources = [("user", text)]
        sources.extend((f"document[{i}]", doc) for i, doc in enumerate(context.documents))
        sources.extend((f"tool[{i}]", out) for i, out in enumerate(context.tool_outputs))
        if (
            any(len(value) > config.max_source_chars for _, value in sources)
            or sum(len(value) for _, value in sources) > config.max_total_chars
        ):
            return self._result(1.0, "input_limit", ["source_length_exceeded"], started)

        peak = 0.0
        reasons: list[str] = []
        degraded = False
        for channel, value in sources:
            score, codes = heuristic_score(value)
            residual = 1.0 - config.heuristic_weight * score
            reasons.extend(f"{channel}.{code}" for code in codes)
            for name, model, gate, weight in self.layers:
                try:
                    if model is None:
                        raise RuntimeError("missing detector")
                    confidence = model(value)
                    if not 0.0 <= confidence <= 1.0:
                        raise ValueError("invalid detector score")
                except Exception:  # noqa: BLE001 - isolate arbitrary provider failures
                    degraded = True
                    reasons.append(f"{channel}.{name}_unavailable")
                    continue
                if confidence >= gate and confidence > 0.0:
                    residual *= 1.0 - weight * confidence
                    reasons.append(f"{channel}.{name}_flagged")
            peak = max(peak, 1.0 - residual)

        category = "prompt_injection"
        if degraded and self.policy.fail_mode == "closed":
            category = "availability"
            peak = 1.0
        return self._result(peak, category, reasons, started)

    def _result(
        self, score: float, category: str, reasons: list[str], started: float
    ) -> GuardResult:
        """Return the decision for standalone or pipeline use, using the same threshold."""
        threshold = self.policy.for_guard(self.name).threshold
        return GuardResult(
            passed=score < (self.policy.threshold if threshold is None else threshold),
            risk_score=score,
            category=category,
            reasons=tuple(reasons),
            latency_ms=(perf_counter() - started) * 1000.0,
        )
