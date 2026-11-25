"""Integration tests for output processing order through the public pipeline."""

import json

from cerberus.core import GuardContext, load_policy
from cerberus.guards.factory import Guardrails


def test_schema_checks_the_released_json() -> None:
    """A schema requiring an email-shaped field must reject the redacted value."""
    rails = Guardrails.build(load_policy())
    original = json.dumps({"contact": "example@example.invalid"})
    context = GuardContext(
        output_schema={
            "type": "object",
            "properties": {"contact": {"type": "string", "pattern": "@"}},
            "required": ["contact"],
        }
    )
    decision = rails.outputs.run(original, context)
    assert decision.blocked and not decision.text
    assert "example.invalid" not in decision.model_dump_json()


def test_valid_redacted_response_can_be_consumed_as_json() -> None:
    """Consumers receive valid sanitized JSON and a stable correlation ID."""
    rails = Guardrails.build(load_policy())
    context = GuardContext(output_schema={"type": "object"})
    decision = rails.outputs.run('{"contact":"sam@example.invalid"}', context)
    assert not decision.blocked
    assert decision.request_id == context.request_id
    assert "@" not in json.loads(decision.text)["contact"]
