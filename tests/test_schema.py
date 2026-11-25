"""Output validation and full-pipeline regeneration regressions."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from cerberus.core.models import Guard, GuardContext, GuardResult
from cerberus.core.pipeline import Pipeline
from cerberus.core.policy import Policy
from cerberus.guards.schema import SchemaGuard, run_with_repair


class Count(BaseModel):
    """A typed application output."""

    count: int


def test_pydantic_and_json_schema_intersection() -> None:
    """Both contracts must accept a value."""
    guard = SchemaGuard(Policy(), Count)
    context = GuardContext(
        output_schema={
            "$defs": {"positive": {"type": "integer", "minimum": 1}},
            "properties": {"count": {"$ref": "#/$defs/positive"}},
        }
    )
    assert guard.check('{"count":1}', context).passed
    assert not guard.check('{"count":0}', context).passed
    assert not guard.check('{"other":1}', context).passed
    assert not guard.check('{"count":"1"}', GuardContext()).passed


@pytest.mark.parametrize("raw", ["oops", '{"x":1,"x":2}', "NaN", "Infinity", "1e5000"])
def test_non_json_has_no_content_in_decision(raw: str) -> None:
    """Malformed JSON and ambiguous extensions cannot pass an empty JSON schema."""
    result = SchemaGuard(Policy()).check(raw, GuardContext(output_schema={}))
    assert not result.passed and result.reasons == ("schema_mismatch",)


def test_unconfigured_and_oversized() -> None:
    """Free-text responses are allowed without a schema; oversized JSON is a hard failure."""
    guard = SchemaGuard(Policy())
    assert guard.check("hi", GuardContext()).passed
    result = guard.check("x" * 65537, GuardContext(output_schema={}))
    assert result.risk_score == 1.0 and result.reasons == ("schema_size_limit",)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "imaginary"},
        {"$schema": "http://json-schema.org/draft-07/schema#"},
        {"properties": {"value": {"$ref": "https://example.invalid/private"}}},
        {"if": {"$dynamicRef": "file:///secret"}},
    ],
)
def test_bad_schema_is_configuration_failure(schema: dict[str, object]) -> None:
    """Unsupported dialects, invalid schemas and external references never trigger repairs."""
    policy = Policy()
    pipeline = Pipeline([SchemaGuard(policy)], policy)
    regenerate = MagicMock()
    result = run_with_repair(pipeline, "{}", GuardContext(output_schema=schema), regenerate, 2)
    assert result.blocked and result.results[-1].category == "availability"
    regenerate.assert_not_called()


class OutputCheck(Guard):
    """Sentinel for other checks and error paths before schema validation."""

    name = "output"

    def __init__(self) -> None:
        """Collect only request IDs for this test."""
        self.ids: list[str] = []

    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Reject private material, sanitize masked material, or accept clean text."""
        self.ids.append(context.request_id)
        if text == "error":
            raise RuntimeError("Upstream outage")
        if text == "private":
            return GuardResult(passed=False, category="private", risk_score=1)
        if text == "mask":
            return GuardResult(passed=False, category="pii", risk_score=0.9, sanitized_text="bad")
        return GuardResult(passed=True, category="ok", risk_score=0)


def test_successful_repair() -> None:
    """All candidates run every output check under one request ID."""
    guard = OutputCheck()
    policy = Policy()
    pipeline = Pipeline([guard, SchemaGuard(policy)], policy)
    context = GuardContext(output_schema={"type": "integer"})
    regenerate = MagicMock(side_effect=["{}", "2"])
    result = run_with_repair(pipeline, "oops", context, regenerate, 2)
    assert not result.blocked and result.text == "2"
    assert guard.ids == [context.request_id] * 3
    assert all(call.args == () and call.kwargs == {} for call in regenerate.call_args_list)


def test_repaired_output_can_be_blocked() -> None:
    """A new sensitive candidate is blocked, and cannot enter another generation."""
    policy = Policy()
    pipeline = Pipeline([OutputCheck(), SchemaGuard(policy)], policy)
    regenerate = MagicMock(return_value="private")
    result = run_with_repair(pipeline, "bad", GuardContext(output_schema={}), regenerate, 3)
    assert result.blocked and result.text == ""
    regenerate.assert_called_once_with()


@pytest.mark.parametrize(
    "text,policy",
    [
        ("bad", Policy(mode="warn")),
        ("error", Policy(fail_mode="open")),
        ("private", Policy()),
        ("mask", Policy(mode="sanitize")),
    ],
)
def test_non_schema_failures_and_warnings_do_not_retry(text: str, policy: Policy) -> None:
    """Never retry after a sensitive or unavailable guard, or an accepted warning."""
    pipeline = Pipeline([OutputCheck(), SchemaGuard(policy)], policy)
    regenerate = MagicMock()
    run_with_repair(pipeline, text, GuardContext(output_schema={}), regenerate, 3)
    regenerate.assert_not_called()


def test_retry_budget_and_order() -> None:
    """Retry budgets are bounded and schema-first short circuits skip regeneration."""
    policy = Policy()
    schema = SchemaGuard(policy)
    context = GuardContext(output_schema={})
    regenerate = MagicMock(return_value="bad")
    result = run_with_repair(Pipeline([schema], policy), "bad", context, regenerate, 2)
    assert result.blocked and result.text == "" and regenerate.call_count == 2
    regenerate.reset_mock()
    run_with_repair(Pipeline([schema, OutputCheck()], policy), "bad", context, regenerate, 1)
    regenerate.assert_not_called()
    with pytest.raises(ValueError):
        run_with_repair(Pipeline([], policy), "", context, regenerate, 4)
