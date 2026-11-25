"""Render content-free offline findings in machine-readable or human-readable form."""

import html
from collections.abc import Iterable, Sequence
from typing import Literal

from cerberus.redteam.evaluator import Metrics, Report

ReportFormat = Literal["json", "markdown", "html"]
_TextFormat = Literal["markdown", "html"]
_MARKDOWN_ESCAPES = {ord(c): f"&#{ord(c)};" for c in "\\&<>|`*_[]!~#:@"}


def _escape(value: str, fmt: _TextFormat) -> str:
    """Render a table cell or inline metadata field as inert text."""
    if fmt == "html":
        return html.escape(value, quote=True)
    return " ".join(value.splitlines()).translate(_MARKDOWN_ESCAPES)


def _table(
    title: str, headers: Sequence[str], rows: Iterable[Sequence[str]], fmt: _TextFormat
) -> str:
    """Build a table with a heading, escaped cells, and semantic HTML headers."""
    if fmt == "html":
        head = "".join(f'<th scope="col">{html.escape(cell)}</th>' for cell in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{_escape(cell, fmt)}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        return (
            f"<h2>{html.escape(title)}</h2>\n"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        )
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(_escape(cell, fmt) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _rate(value: float | None) -> str:
    """Distinguish unmeasured rates from observed zero."""
    return "Unmeasured" if value is None else f"{value:.1%}"


def _guard_row(name: str, values: Metrics) -> tuple[str, ...]:
    """Summarize guard results and the counts excluded for availability."""
    return (
        name,
        str(values.cases),
        str(values.positive_cases - values.unavailable_positive),
        str(values.guard_misses),
        _rate(values.guard_miss_rate),
        str(values.benign_cases - values.unavailable_benign),
        str(values.false_positives),
        _rate(values.false_positive_rate),
        str(values.unavailable),
    )


def _recorded_row(name: str, values: Metrics) -> tuple[str, ...]:
    """Summarize coverage and outcomes of independently supplied response recordings."""
    return (
        name,
        str(values.recorded_eligible),
        str(values.recorded_observed),
        str(values.recorded_eligible - values.recorded_observed),
        str(values.recorded_leaks),
        _rate(values.recorded_leak_rate),
    )


def render_report(report: Report, fmt: ReportFormat) -> str:
    """Export JSON or a standalone document with category metrics and ranked findings."""
    if fmt == "json":
        return report.model_dump_json(indent=2)
    if fmt not in {"markdown", "html"}:
        raise ValueError("unsupported report format")

    report_id = _escape(report.id, fmt)
    suite = _escape(report.caseset_version, fmt)
    metadata = (
        f"Run {report_id} · Fixture version {suite} · "
        f"Offline heuristics · {_escape(report.mode, fmt)} mode · "
        f"Input guard {'enabled' if report.input_enabled else 'disabled'} · "
        f"Weight {report.heuristic_weight:g} · Threshold {report.threshold:g} · "
        f"Critical threshold {report.critical_threshold:g}"
    )
    groups = [("All cases", report.totals), *sorted(report.categories.items())]
    guard = _table(
        "Local guard decisions",
        (
            "Category",
            "Cases",
            "Scored positive",
            "Missed",
            "Miss rate",
            "Scored benign",
            "False positives",
            "False-positive rate",
            "Unavailable",
        ),
        (_guard_row(name, values) for name, values in groups),
        fmt,
    )
    recorded = _table(
        "Recorded application responses",
        ("Category", "Eligible", "Observed", "Unobserved", "Canary leaks", "Observed leak rate"),
        (_recorded_row(name, values) for name, values in groups),
        fmt,
    )
    if report.findings:
        findings = _table(
            "Findings by severity",
            ("Severity", "Case", "Category", "Finding", "Recommended mitigation"),
            (
                (item.severity, item.case_id, item.category, item.kind, item.mitigation)
                for item in report.findings
            ),
            fmt,
        )
    else:
        findings = (
            "<h2>Findings by severity</h2><p>No findings in scored observations.</p>"
            if fmt == "html"
            else "## Findings by severity\n\nNo findings in scored observations."
        )
    guide = (
        "Guard rates use scored cases of the corresponding label. "
        "The observed leak rate uses eligible cases with a supplied recording. "
        "Unmeasured means the denominator is zero."
    )
    if fmt == "markdown":
        return (
            f"# Cerberus offline evaluation\n\n{metadata}\n\n{guide}\n\n"
            f"{guard}\n\n{recorded}\n\n{findings}"
        )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Cerberus offline evaluation</title>"
        "<style>body{font:1rem/1.5 system-ui,sans-serif;max-width:84rem;margin:2rem auto;"
        "padding:0 1rem;color:#182124}table{border-collapse:collapse;display:block;"
        "overflow-x:auto}th,td{text-align:left;vertical-align:top;padding:.45rem .6rem;"
        "border-bottom:1px solid #bbc6ca}th{background:#eaf0f2}</style></head><body>"
        f"<main><h1>Cerberus offline evaluation</h1><p>{metadata}</p><p>{guide}</p>"
        f"{guard}\n{recorded}\n{findings}</main></body></html>"
    )
