"""Report rendering contracts, including missing observations and escaping."""

import json
from typing import Any

import pytest

from cerberus.core.policy import load_policy
from cerberus.redteam.evaluator import Finding, Report, evaluate
from cerberus.redteam.export import render_report


@pytest.fixture
def report() -> Report:
    """Compute a local report carrying exactly one deliberately observed leak."""
    return evaluate(load_policy(), {"direct-scoreboard": "CERBERUS_CANARY_SCOREBOARD"})


def test_exported_rates_and_measurement_names(report: Report) -> None:
    """Serialize the derived denominators and label conditional rates in display formats."""
    raw = json.loads(render_report(report, "json"))
    assert raw["totals"]["recorded_leak_rate"] == 1.0
    assert raw["totals"]["recorded_observed"] == 1
    assert "CERBERUS_CANARY_SCOREBOARD" not in str(raw)
    md = render_report(report, "markdown")
    assert "Local guard decisions" in md
    assert "Recorded application responses" in md
    assert "Unmeasured" in md and "100.0%" in md
    assert "Recommended mitigation" in md
    assert "CERBERUS_CANARY_SCOREBOARD" not in md
    page = render_report(report, "html")
    assert page.startswith("<!doctype html>")
    assert "<table>" in page
    assert "CERBERUS_CANARY_SCOREBOARD" not in page


def test_untrusted_values_cannot_inject_markup(report: Report) -> None:
    """Neutralize tags, link syntax, newlines and column separators in rendered fields."""
    hostile = "<img src=x>|[link](x)\n# new"
    item = Finding(
        case_id=hostile,
        category=hostile,
        severity="low",
        kind="guard_miss",
        mitigation=hostile,
    )
    changed = report.model_copy(
        update={"id": hostile, "categories": {hostile: report.totals}, "findings": (item,)}
    )
    md = render_report(changed, "markdown")
    assert "<img" not in md and "[link]" not in md and "\n# new" not in md
    assert "&#124;" in md
    page = render_report(changed, "html")
    assert "<img" not in page and "&lt;img" in page


def test_empty_findings_label(report: Report) -> None:
    """An empty finding section remains interpretable in either text format."""
    clean = report.model_copy(update={"findings": ()})
    assert "No findings in scored observations." in render_report(clean, "markdown")
    assert "No findings in scored observations." in render_report(clean, "html")


def test_reject_unknown_format(report: Report) -> None:
    """Runtime callers receive a clear error for unsupported format names."""
    supplied: Any = "xml"
    with pytest.raises(ValueError, match="format"):
        render_report(report, supplied)
