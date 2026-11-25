"""Validated, administrator-controlled guardrail policies."""

from importlib import resources
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

Score = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class PolicySettings(BaseModel):
    """Common validation rules for policy files and their sections."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class GuardPolicy(PolicySettings):
    """Controls one named guard and optionally overrides its risk threshold."""

    enabled: bool = True
    threshold: Score | None = None


_DEFAULT_GUARD_POLICY = GuardPolicy()


class InputSettings(PolicySettings):
    """Input detector weights, activation gates, and per-request workload limits."""

    ml_enabled: bool = False
    judge_enabled: bool = False
    heuristic_weight: Score = 1.0
    ml_weight: Score = 0.9
    judge_weight: Score = 0.8
    ml_threshold: Score = 0.5
    judge_threshold: Score = 0.5
    max_sources: int = Field(default=16, ge=1, le=32)
    max_source_chars: int = Field(default=32_768, ge=1, le=32_768)
    max_total_chars: int = Field(default=65_536, ge=1, le=131_072)


class OutputSettings(PolicySettings):
    """Output filtering and verification options, independent of provider credentials."""

    pii_mode: Literal["redact", "mask", "hash"] = "redact"
    ner_enabled: bool = False
    ner_model: Annotated[str, Field(min_length=1)] = "en_core_web_sm"
    max_retries: int = Field(default=1, ge=0, le=3)
    grounding_enabled: bool = False
    grounding_threshold: Score = 0.65
    max_chars: int = Field(default=32_768, ge=1, le=131_072)


class Policy(PolicySettings):
    """Defines the action, availability behavior, and boundaries for a pipeline."""

    mode: Literal["block", "warn", "sanitize"] = "block"
    fail_mode: Literal["open", "closed"] = "closed"
    threshold: Score = 0.65
    critical_threshold: Score = 0.95
    guards: dict[Annotated[str, Field(min_length=1)], GuardPolicy] = Field(default_factory=dict)
    input: InputSettings = Field(default_factory=InputSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> Self:
        """Keep every regular risk threshold at or below the critical threshold."""
        if self.threshold > self.critical_threshold:
            raise ValueError("threshold exceeds critical_threshold")
        for name, guard in self.guards.items():
            if guard.threshold is not None and guard.threshold > self.critical_threshold:
                raise ValueError(f"threshold for {name!r} exceeds critical_threshold")
        return self

    def for_guard(self, name: str) -> GuardPolicy:
        """Return the named override, or a configuration using pipeline defaults."""
        return self.guards.get(name, _DEFAULT_GUARD_POLICY)


class _PolicyLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects ambiguous duplicate mapping keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[object, object]:
        """Parse a YAML mapping and report duplicate keys with their locations."""
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) != len(node.value):
            raise ConstructorError(
                "while constructing a policy",
                node.start_mark,
                "duplicate mapping key",
                node.end_mark,
            )
        return mapping


def load_policy(source: str | Path = "strict") -> Policy:
    """Load the packaged strict policy or a trusted administrator's YAML file."""
    if source == "strict":
        content = resources.files("cerberus.policies").joinpath("strict.yaml").read_text("utf-8")
    else:
        content = Path(source).read_text(encoding="utf-8")
    return Policy.model_validate(yaml.load(content, Loader=_PolicyLoader))
