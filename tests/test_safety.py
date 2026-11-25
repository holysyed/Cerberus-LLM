"""Safety and grounding checks on synthetic outputs and ordinary benign language."""

import pytest

from cerberus.core.models import GuardContext
from cerberus.core.pipeline import Pipeline
from cerberus.core.policy import GuardPolicy, InputSettings, OutputSettings, Policy
from cerberus.guards.safety import GroundingGuard, SafetyGuard


@pytest.mark.parametrize(
    ("output", "category"),
    [
        ("You're a complete idiot.", "toxicity"),
        ("I will physically hurt you.", "unsafe_content"),
        ("You should hurt yourself.", "unsafe_content"),
        ("I can't share this. However, here is the private fixture.", "refusal_bypass"),
    ],
)
def test_safety_signals(output: str, category: str) -> None:
    """Recognize explicit synthetic examples in each supported behavior class."""
    result = SafetyGuard(Policy()).check(output, GuardContext())
    assert not result.passed
    assert result.category == category
    assert output not in result.model_dump_json()


@pytest.mark.parametrize(
    "output",
    [
        "I cannot share private information. Here is the public summary.",
        "However, here are the private notes you wrote yesterday.",
        "The unit test checks for a word like idiot.",
        "I will help you find assistance.",
        "The documentation explains the term system prompt.",
    ],
)
def test_benign_outputs(output: str) -> None:
    """Refusals, ordinary text and general discussions should pass."""
    assert SafetyGuard(Policy()).check(output, GuardContext()).passed


@pytest.mark.parametrize(
    ("system", "output"),
    [
        ("SYNTHETIC CANARY", "A sample labeled Synthetic-Canary appeared."),
        (
            "Use the unique synthetic orchard calendar for orange ledger entries today",
            "The unique synthetic orchard calendar for orange ledger entries.",
        ),
    ],
)
def test_prompt_disclosure_is_critical(system: str, output: str) -> None:
    """Suppress prompt fragments even in warning mode."""
    policy = Policy(mode="warn")
    result = Pipeline([SafetyGuard(policy)], policy).run(output, GuardContext(system_prompt=system))
    assert result.blocked and not result.text
    finding = result.results[0]
    assert finding.risk_score == 1.0
    assert finding.reasons == ("system_prompt_overlap",)


def test_short_common_prompt_is_not_a_leak() -> None:
    """A short greeting cannot establish system prompt disclosure."""
    assert SafetyGuard(Policy()).check("Hello there.", GuardContext(system_prompt="Hello")).passed


def test_safety_reads_json_strings() -> None:
    """Inspect escaped JSON fields including deeply nested generated values."""
    guard = SafetyGuard(Policy())
    context = GuardContext(system_prompt="SYNTHETIC CANARY")
    output = r'{"items":[3,{"answer":"SYNTHETIC\u0020CANARY"}]}'
    assert guard.check(output, context).category == "prompt_leakage"
    assert guard.check(r'"You\u0020are an idiot"', context).category == "toxicity"
    assert guard.check(r"[invalid\u0020json", context).passed


def test_guard_override_and_size_bounds() -> None:
    """Apply global resource bounds and the configured safety threshold."""
    policy = Policy(guards={"safety": GuardPolicy(threshold=0.94)})
    assert SafetyGuard(policy).check("You are an idiot.", GuardContext()).passed
    small = SafetyGuard(Policy(output=OutputSettings(max_chars=8)))
    assert not small.check("x" * 9, GuardContext()).passed
    assert not small.check("ok", GuardContext(system_prompt="x" * 9)).passed


def test_grounding_opt_in() -> None:
    """Grounding is enabled through policy and supports document and tool evidence."""
    assert GroundingGuard(Policy()).check("A claim", GuardContext()).passed
    guard = GroundingGuard(Policy(output=OutputSettings(grounding_enabled=True)))
    context = GuardContext(documents=("Mars is red.",), tool_outputs=("Samples arrived in 2023.",))
    assert guard.check("Mars is red. Samples arrived in 2023.", context).passed
    result = guard.check("Mars is red. Samples arrived in 2024.", context)
    assert not result.passed
    assert result.reasons == ("unsupported_number",)
    assert not guard.check("Copper conducts electricity.\nMars is red.", context).passed
    assert guard.check("When did the samples arrive? The.", context).passed
    assert not guard.check("Stars rotate.", GuardContext()).passed


def test_grounding_resource_limits() -> None:
    """Enforce source count, evidence size, and output size before lexical work."""
    guard = GroundingGuard(
        Policy(
            input=InputSettings(max_sources=2, max_total_chars=5),
            output=OutputSettings(grounding_enabled=True, max_chars=10),
        )
    )
    assert guard.check("Moon", GuardContext(documents=("a", "b"))).risk_score == 1.0
    assert guard.check("Moon", GuardContext(documents=("abcdef",))).risk_score == 1.0
    assert guard.check("a" * 11, GuardContext()).risk_score == 1.0


def test_long_unpunctuated_line() -> None:
    """Evaluate complete long lines ending in a newline as their own claim."""
    guard = GroundingGuard(Policy(output=OutputSettings(grounding_enabled=True)))
    decision = guard.check(
        "quartz " * 4000 + "\nMars is red.", GuardContext(documents=("Mars is red.",))
    )
    assert not decision.passed
