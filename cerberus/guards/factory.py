"""Construct the standard input and output pipelines for one application policy."""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from cerberus.core.audit import AuditStore
from cerberus.core.pipeline import Pipeline
from cerberus.core.policy import Policy
from cerberus.guards.injection import InputGuard
from cerberus.guards.pii import PiiGuard
from cerberus.guards.safety import GroundingGuard, SafetyGuard
from cerberus.guards.schema import SchemaGuard


@dataclass(frozen=True)
class Guardrails:
    """Application-scoped pipelines sharing one policy and optional audit database."""

    inputs: Pipeline
    outputs: Pipeline

    @classmethod
    def build(
        cls,
        policy: Policy,
        *,
        audit_store: AuditStore | None = None,
        hash_key: bytes | None = None,
        classifier: Callable[[str], float] | None = None,
        judge: Callable[[str], float] | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> "Guardrails":
        """Check safety before releasing sanitized text, then verify its final schema."""
        return cls(
            Pipeline([InputGuard(policy, classifier, judge)], policy, audit_store),
            Pipeline(
                [
                    SafetyGuard(policy),
                    GroundingGuard(policy),
                    PiiGuard(policy, hash_key=hash_key),
                    SchemaGuard(policy, model=output_model),
                ],
                policy,
                audit_store,
            ),
        )
