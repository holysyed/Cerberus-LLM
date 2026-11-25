"""Output contract validation and bounded regeneration of malformed responses."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from functools import lru_cache
from time import perf_counter
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ValidationError
from referencing import Registry
from referencing.jsonschema import DRAFT202012

from cerberus.core.models import Guard, GuardContext, GuardResult, PipelineResult
from cerberus.core.pipeline import Pipeline
from cerberus.core.policy import Policy

_MAX_JSON_CHARS = 65_536
_DIALECT = "https://json-schema.org/draft/2020-12/schema"


@lru_cache(maxsize=32)
def _validator(schema_json: str) -> Draft202012Validator:
    """Compile an administrator's schema with fragment-only references."""
    schema = json.loads(schema_json)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        raise ValueError("Invalid application output schema") from None
    pending = [DRAFT202012.create_resource(schema)]
    while pending:
        resource = pending.pop()
        node = resource.contents
        if isinstance(node, dict):
            if node.get("$schema", _DIALECT).rstrip("#") != _DIALECT:
                raise ValueError("Only JSON Schema draft 2020-12 is supported")
            for keyword in ("$ref", "$dynamicRef"):
                reference = node.get(keyword, "")
                if reference and not reference.startswith("#"):
                    raise ValueError("Only local schema references are supported")
            pending.extend(resource.subresources())
    return Draft202012Validator(
        schema, registry=Registry(), format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def _finite_number(raw: str) -> float:
    """Reject NaN, Infinity, and numbers exceeding the supported float range."""
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Non-finite JSON number")
    return value


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Require unambiguous object member names."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON property")
    return result


class SchemaGuard(Guard):
    """Apply the context's JSON Schema and an optional application Pydantic model."""

    name = "schema"

    def __init__(self, policy: Policy, model: type[BaseModel] | None = None) -> None:
        """Bind application-owned policy and validation model."""
        self._model = model
        threshold = policy.for_guard(self.name).threshold
        self._threshold = threshold if threshold is not None else policy.threshold

    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Validate JSON without including values or validator diagnostics in decisions."""
        start = perf_counter()
        reason = None
        score = 0.0
        if context.output_schema is not None or self._model is not None:
            if len(text) > _MAX_JSON_CHARS:
                score, reason = 1.0, "schema_size_limit"
            elif not self._valid(text, context.output_schema):
                score, reason = 0.9, "schema_mismatch"
        return GuardResult(
            passed=score < self._threshold,
            category=self.name,
            risk_score=score,
            reasons=(reason,) if reason else (),
            latency_ms=(perf_counter() - start) * 1000,
        )

    def _valid(self, text: str, schema: dict[str, Any] | None) -> bool:
        """Validate trusted schemas separately from model-generated values."""
        validator = _validator(json.dumps(schema, sort_keys=True)) if schema is not None else None
        try:
            data = json.loads(
                text,
                object_pairs_hook=_object,
                parse_float=_finite_number,
                parse_constant=_finite_number,
            )
        except (ValueError, RecursionError):
            return False
        if validator is not None and not validator.is_valid(data):
            return False
        if self._model is not None:
            try:
                self._model.model_validate_json(text, strict=True)
            except ValidationError:
                return False
        return True


def run_with_repair(
    pipeline: Pipeline,
    text: str,
    context: GuardContext,
    regenerate: Callable[[], str],
    max_retries: int,
) -> PipelineResult:
    """Regenerate only when every enabled check ran and only the schema blocked.

    Applications should put SchemaGuard last. The callback receives no rejected
    output or validation diagnostics and should use the original guarded prompt.
    Every candidate gets the same request ID and runs the complete pipeline.
    """
    if type(max_retries) is not int or not 0 <= max_retries <= 3:
        raise ValueError("max_retries must be an integer between 0 and 3")
    enabled = [
        guard.name for guard in pipeline.guards if pipeline.policy.for_guard(guard.name).enabled
    ]
    result = pipeline.run(text, context)
    for _ in range(max_retries):
        if not result.blocked or len(result.results) != len(enabled):
            break
        last = result.results[-1]
        if (
            enabled[-1] != SchemaGuard.name
            or last.category != SchemaGuard.name
            or last.reasons != ("schema_mismatch",)
            or any(
                not item.passed or item.category == "availability" for item in result.results[:-1]
            )
        ):
            break
        result = pipeline.run(regenerate(), context)
    return result
