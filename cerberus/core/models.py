"""Contracts shared by guards, the gateway, and local integrations."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _new_request_id() -> str:
    """Generate a correlation ID without relying on caller-supplied data."""
    return str(uuid4())


class GuardContext(BaseModel):
    """Trusted call metadata and application-supplied evidence for guard checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(default_factory=_new_request_id)
    documents: tuple[str, ...] = ()
    tool_outputs: tuple[str, ...] = ()
    system_prompt: str | None = None
    output_schema: dict[str, Any] | None = None

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        """Canonicalize the ID and prevent arbitrary text entering audit metadata."""
        return str(UUID(value))


class GuardResult(BaseModel):
    """One decision, with stable reason codes and an optional safe replacement."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    passed: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    category: str = Field(min_length=1, max_length=80)
    reasons: tuple[str, ...] = ()
    sanitized_text: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)


class Guard(ABC):
    """A synchronous detector that evaluates a single input or output."""

    name: str

    @abstractmethod
    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Evaluate text and return a score, metadata, and any safe replacement."""


class PipelineResult(BaseModel):
    """The final releasable text and decisions in execution order."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    request_id: str
    blocked: bool
    text: str
    results: tuple[GuardResult, ...]
    latency_ms: float = Field(ge=0.0)
