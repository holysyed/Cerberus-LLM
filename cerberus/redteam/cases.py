"""Fixed, versioned cases for local guards and previously recorded responses."""

from functools import lru_cache
from importlib import resources
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CASESET_VERSION = "1.0.0"
Category = Literal[
    "direct_injection",
    "jailbreak",
    "prompt_leaking",
    "pii_extraction",
    "indirect_injection",
    "encoding_obfuscation",
    "multi_turn",
    "benign_control",
]
Severity = Literal["low", "medium", "high", "critical"]
Sample = Annotated[str, Field(min_length=1, max_length=8192)]


class Case(BaseModel):
    """One fixed observation, its guard expectation, and optional disclosure marker."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    category: Category
    severity: Severity
    text: Sample
    documents: tuple[Sample, ...] = Field(default=(), max_length=4)
    tool_outputs: tuple[Sample, ...] = Field(default=(), max_length=4)
    expected_block: bool
    canary: str | None = Field(default=None, pattern=r"^CERBERUS_CANARY_[A-Z0-9_]{4,40}$")
    mitigation: str = Field(min_length=1, max_length=360)

    @model_validator(mode="after")
    def check_canary_separation(self) -> Self:
        """Keep disclosure markers distinct from material a model can safely quote."""
        if self.canary and any(
            self.canary in source for source in (self.text, *self.documents, *self.tool_outputs)
        ):
            raise ValueError("Disclosure marker appears in untrusted fixture data")
        if self.category == "benign_control" and (self.expected_block or self.canary):
            raise ValueError("A benign control must allow passage and omit the disclosure marker")
        return self


class _CasesFile(BaseModel):
    """Envelope for the packaged case collection."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["1.0.0"]
    cases: tuple[Case, ...] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def verify_unique_keys(self) -> Self:
        """Require distinct sample IDs and record-specific disclosure markers."""
        ids = [case.id for case in self.cases]
        markers = [case.canary for case in self.cases if case.canary]
        if len(set(ids)) != len(ids) or len(set(markers)) != len(markers):
            raise ValueError("Duplicate case ID or disclosure marker")
        return self


@lru_cache(maxsize=1)
def load_cases() -> tuple[Case, ...]:
    """Load and validate the packaged fixture library without running any inference."""
    resource = resources.files("cerberus.redteam").joinpath("cases.v1.json")
    return _CasesFile.model_validate_json(resource.read_bytes()).cases
