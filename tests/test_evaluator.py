"""Check local evaluation accounting, isolation, and recording confidentiality."""

import pytest

from cerberus.core.models import GuardContext, GuardResult
from cerberus.core.policy import GuardPolicy, InputSettings, Policy
from cerberus.guards.injection import InputGuard
from cerberus.redteam.cases import load_cases
from cerberus.redteam.evaluator import Report, evaluate


def test_counts_order_and_round_trip() -> None:
    """Persisted JSON retains all computed metrics and findings in severity order."""
    cases = load_cases()
    report = evaluate(Policy())
    assert Report.model_validate(report.model_dump(mode="json")) == report
    assert report.totals.cases == len(cases)
    assert report.totals.positive_cases == sum(case.expected_block for case in cases)
    assert sum(group.cases for group in report.categories.values()) == len(cases)
    assert report.totals.recorded_eligible == sum(case.canary is not None for case in cases)
    assert report.totals.recorded_observed == report.totals.unavailable == 0
    assert report.totals.recorded_leak_rate is None
    assert report.totals.false_positive_rate == 0
    ranks = ["critical", "high", "medium", "low"]
    assert list(report.findings) == sorted(
        report.findings, key=lambda f: (ranks.index(f.severity), f.case_id, f.kind)
    )
    assert report.categories["benign_control"].guard_miss_rate is None


def test_recordings_are_independent_and_private() -> None:
    """Observe leaks solely in eligible recordings and exclude all raw content from reports."""
    positives = [case for case in load_cases() if case.canary]
    first, second = positives[:2]
    assert first.canary and second.canary
    benign = next(case for case in load_cases() if case.category == "benign_control")
    report = evaluate(
        Policy(),
        {
            first.id: "private fixture response " + first.canary,
            second.id: "",
            benign.id: first.canary,
        },
    )
    assert report.totals.recorded_observed == 2
    assert report.totals.recorded_leaks == 1
    assert report.totals.recorded_leak_rate == 0.5
    assert report.categories["benign_control"].recorded_leak_rate is None
    assert [f.case_id for f in report.findings if f.kind == "recorded_canary_leak"] == [first.id]
    payload = report.model_dump_json()
    assert "private fixture response" not in payload and first.canary not in payload
    assert first.text not in payload


def test_warn_and_disabled_input_policy_score_enforcement() -> None:
    """Disabling enforcement allows expected blocks and is visible in the report."""
    disabled = evaluate(Policy(guards={"input": GuardPolicy(enabled=False)}))
    assert not disabled.input_enabled and disabled.totals.guard_miss_rate == 1
    warned = evaluate(Policy(mode="warn"))
    assert warned.totals.guard_misses > 0 and warned.totals.unavailable == 0


def test_backend_failures_are_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both fail-open and fail-closed paths have explicit, non-success outcomes."""

    def fail(self: InputGuard, text: str, context: GuardContext) -> GuardResult:
        """Simulate operational failure with no dependency on a live backend."""
        raise RuntimeError

    monkeypatch.setattr(InputGuard, "check", fail)
    for policy in [Policy(), Policy(fail_mode="open")]:
        report = evaluate(policy)
        assert report.totals.unavailable == report.totals.cases
        assert report.totals.false_positive_rate is None
        assert report.totals.guard_miss_rate is None
        assert report.totals.guard_misses == report.totals.false_positives == 0
        assert {f.kind for f in report.findings} == {"guard_unavailable"}


def test_positive_false_positives_and_no_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local evaluation never initializes configured online-capable layers."""

    def forbidden() -> None:
        """Fail immediately if a Hugging Face backend is constructed."""
        pytest.fail("Unexpected model construction")

    monkeypatch.setattr("cerberus.guards.injection.HFClassifier", forbidden)
    policy = Policy(threshold=0.0, input=InputSettings(ml_enabled=True, judge_enabled=True))
    result = evaluate(policy)
    assert result.totals.false_positive_rate == 1 and result.totals.guard_miss_rate == 0
    assert result.totals.false_positives == result.totals.benign_cases
    assert any(f.kind == "false_positive" for f in result.findings)
    assert policy.input.ml_enabled and policy.input.judge_enabled


def test_recordings_must_be_bounded_and_known() -> None:
    """Reject unknown recording references and overlong values."""
    with pytest.raises(ValueError, match="unknown fixture"):
        evaluate(Policy(), {"unknown": "sample"})
    with pytest.raises(ValueError, match="65,536"):
        evaluate(Policy(), {load_cases()[0].id: "a" * 65_537})
