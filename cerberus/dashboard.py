"""Read-only Streamlit views of live guard decisions and saved offline evaluations."""

from __future__ import annotations

import os
import sys
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from statistics import median

import streamlit as st
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from cerberus.core.audit import AuditStore, RequestActivity
from cerberus.redteam.evaluator import Report
from cerberus.redteam.export import ReportFormat, render_report

STYLE = """
<style>
.stMainBlockContainer {max-width:1320px; padding-top:2.3rem; padding-bottom:2rem;}
header[data-testid="stHeader"] {background:transparent; height:2rem;}
.c-brand {display:flex;gap:12px;align-items:center;padding:0 0 17px;
 border-bottom:1px solid #cdd7cf;font:600 25px/1.1 'Inter',sans-serif;letter-spacing:-.9px}
.c-brand small {border-left:1px solid #c3cec6;padding-left:13px;margin-left:2px;
 font:500 13px/1.5 'Inter',sans-serif;letter-spacing:0;color:#4e685b}
.c-mark {display:inline-flex;gap:3px;align-items:center;justify-content:center;width:36px;height:36px;
 border-radius:10px;background:#114f41;transform:rotate(-2deg)}
.c-mark i {height:15px;width:4px;border-radius:2px;background:#cbf5cc;transform:skewY(-25deg)}
.c-mark i:nth-child(2){height:23px}
.c-heading {font:500 clamp(32px,4vw,48px)/1.12 Georgia,serif;letter-spacing:-2px;
 color:#183b30;margin:2px 0 5px}
.c-note,.c-label,.c-row,.c-foot {font:500 13px/1.55 'Inter',sans-serif;color:#52645a}
.c-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:15px 0 1px}
.c-card,.c-panel {border:1px solid #d6ded6;border-radius:12px;background:#fff;padding:17px 19px}
.c-card strong {display:block;font:550 clamp(26px,3vw,39px)/1.25 'Inter',sans-serif;
 font-variant-numeric:tabular-nums;letter-spacing:-1.7px;color:#183b30}
.c-card .c-foot {font-size:12px;color:#657368}
.c-panels {display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:3px 0}
.c-panel h3,.c-logtitle {font:600 18px/1.4 'Inter',sans-serif;letter-spacing:-.45px;color:#243d30;margin:0 0 11px}
.c-bar {height:13px;display:flex;overflow:hidden;border-radius:5px;background:#edf1ed;margin:13px 0}
.c-bar i {background:#247559}.c-bar b {background:#bb5545}
.c-row {display:flex;justify-content:space-between;align-items:center;gap:12px;margin:5px 0}
.c-row em {flex:1;background:#eef2ed;border-radius:3px;height:5px}
.c-row em i {display:block;height:5px;background:#548e79;border-radius:3px}
.c-scroll {max-height:430px;overflow:auto;border:1px solid #d6ded6;border-radius:10px;background:#fff}
.c-scroll table {width:100%;border-collapse:collapse;white-space:nowrap;font:13px/1.5 'Inter',sans-serif}
.c-scroll th,.c-scroll td {text-align:left;padding:9px 12px;border:0;border-bottom:1px solid #e6ebe5}
.c-scroll th {position:sticky;top:0;background:#eef2ec;color:#506353;font-size:12px;font-weight:600}
.c-scroll td code {font:12px Consolas,monospace;color:#486251;background:transparent}
.c-pill {border-radius:12px;padding:3px 9px;background:#e4f3ea;color:#226147;font-size:12px;font-weight:600}
.c-pill.blocked {background:#f8e9e4;color:#943f32}
@media (max-width:750px){.c-grid{grid-template-columns:repeat(2,1fr)}.c-panels{grid-template-columns:1fr}}
</style>
"""


def cards(items: list[tuple[str, str, str]]) -> None:
    """Render metrics and their units or denominators together."""
    cells = [
        f'<div class="c-card"><div class="c-label">{escape(label)}</div>'
        f'<strong>{escape(value)}</strong><div class="c-foot">{escape(note)}</div></div>'
        for label, value, note in items
    ]
    st.html('<div class="c-grid">' + "".join(cells) + "</div>")


def heading(title: str, note: str) -> None:
    """Display a view heading and its scope."""
    st.html(f'<h1 class="c-heading">{escape(title)}</h1><div class="c-note">{escape(note)}</div>')


def show_activity(store: AuditStore, limit: int) -> None:
    """Show correlated requests with their latest complete guard decision."""
    rows = store.request_activity(limit)
    heading("Request activity", f"Latest {limit} guarded requests · Times in UTC")
    if not rows:
        st.info("No guarded requests yet. Run a guard or chat request against the gateway.")
        return
    blocked = sum(r.blocked for r in rows)
    with_pii = sum(bool(r.pii_types) for r in rows)
    injection = sum(r.block_category == "prompt_injection" for r in rows)
    cards(
        [
            ("Guarded requests", str(len(rows)), "Distinct request IDs"),
            ("Blocked requests", str(blocked), f"{injection} prompt injection blocks"),
            ("Requests with PII", str(with_pii), "Includes detected secrets"),
            (
                "Median guard time",
                f"{median(r.guard_ms for r in rows):.2f} ms",
                "Completed stages per request",
            ),
        ]
    )
    allowed = len(rows) - blocked
    count = Counter(kind for row in rows for kind in row.pii_types)
    labels = {"pii": "PII", "ifsc": "IFSC", "pan": "PAN", "ssn": "SSN"}
    bars = "".join(
        f'<div class="c-row"><span>{escape(labels.get(kind, kind.title()))}</span>'
        f'<em><i style="width:{100 * n / max(count.values()):.1f}%"></i></em><span>{n}</span></div>'
        for kind, n in count.most_common()
    )
    allowed_pct = 100 * allowed / len(rows)
    st.html(
        f'<div class="c-panels"><section class="c-panel"><h3>Guard decisions</h3>'
        f'<div class="c-bar"><i style="width:{allowed_pct:.2f}%"></i>'
        f'<b style="flex:1"></b></div>'
        f'<div class="c-row"><span>Allowed</span><span>{allowed} · {allowed_pct:.0f}%</span></div>'
        f'<div class="c-row"><span>Blocked</span><span>{blocked} · {100 - allowed_pct:.0f}%</span></div>'
        f'</section><section class="c-panel"><h3>Detected data types</h3>'
        f"{bars or '<div class=c-note>No typed detections in this window.</div>'}</section></div>"
    )
    st.html('<h2 class="c-logtitle">Request log</h2>')
    selected = st.radio(
        "Filter by guard decision",
        ["All", "Blocked", "Allowed"],
        horizontal=True,
        label_visibility="collapsed",
        key="activity-outcome",
    )
    filtered = [r for r in rows if selected == "All" or (selected == "Blocked") == r.blocked]
    if filtered:
        st.html(
            '<div class="c-scroll"><table><thead><tr>'
            "<th>Time (UTC)</th><th>Decision</th><th>Stage</th><th>Finding</th>"
            "<th>Peak risk</th><th>Guard time</th><th>Request ID</th></tr></thead><tbody>"
            + "".join(log_row(row) for row in filtered)
            + "</tbody></table></div>"
        )
    else:
        st.caption("No requests match this filter.")


def log_row(row: RequestActivity) -> str:
    """Escape all stored metadata before presenting one row of the request log."""
    date = datetime.fromisoformat(row.timestamp).astimezone(UTC).strftime("%b %d %H:%M:%S")
    outcome = "Blocked" if row.blocked else "Allowed"
    finding = (
        row.block_category.replace("_", " ") if row.blocked else ", ".join(sorted(row.pii_types))
    )
    return (
        f'<tr><td>{escape(date)}</td><td><span class="c-pill {outcome.lower()}">{outcome}</span></td>'
        f"<td>{escape(' + '.join(sorted(row.stages)) or '—')}</td><td>{escape(finding or '—')}</td>"
        f"<td>{row.risk:.2f}</td><td>{row.guard_ms:.2f} ms</td><td><code>{escape(row.request_id)}</code></td></tr>"
    )


def show_reports(store: AuditStore) -> None:
    """Inspect a saved offline run and download its existing export formats."""
    heading("Evaluation reports", "Saved offline evaluations · Fixed synthetic fixtures")
    recent = store.recent_reports(100)
    if not recent:
        st.info("No saved reports yet. Use POST /v1/redteam/run or run the local demo.")
        return
    labels = {r["id"]: f"{r['timestamp'][:19].replace('T', ' ')} UTC · {r['id']}" for r in recent}
    choice = st.selectbox("Saved run", list(labels), format_func=labels.__getitem__)
    raw = store.get_report(choice)
    if raw is None:
        st.warning("This run is unavailable.")
        return
    try:
        report = Report.model_validate(raw)
    except ValidationError:
        st.warning("This saved report has an unsupported format.")
        return
    totals = report.totals
    cards(
        [
            ("Fixtures", str(totals.cases), f"Suite {report.caseset_version}"),
            (
                "Guard misses",
                str(totals.guard_misses),
                f"{totals.positive_cases - totals.unavailable_positive} scored positives",
            ),
            (
                "False positives",
                str(totals.false_positives),
                f"{totals.benign_cases - totals.unavailable_benign} scored benign",
            ),
            (
                "Observed canary leaks",
                str(totals.recorded_leaks) if totals.recorded_observed else "—",
                f"{totals.recorded_observed} recordings observed",
            ),
        ]
    )
    st.subheader("Findings by severity")
    if report.findings:
        st.dataframe(
            [
                {
                    "Severity": f.severity.title(),
                    "Case": f.case_id,
                    "Finding": f.kind.replace("_", " "),
                    "Mitigation": f.mitigation,
                }
                for f in report.findings
            ],
            hide_index=True,
        )
    else:
        st.caption("No findings for this run.")
    with st.expander("Category scores"):
        st.dataframe(
            [
                {
                    "Category": k.replace("_", " "),
                    "Cases": v.cases,
                    "Miss rate": v.guard_miss_rate,
                    "False-positive rate": v.false_positive_rate,
                    "Unavailable": v.unavailable,
                    "Recorded leak rate": v.recorded_leak_rate,
                }
                for k, v in report.categories.items()
            ],
            hide_index=True,
        )
    formats: list[tuple[ReportFormat, str, str]] = [
        ("json", "json", "application/json"),
        ("markdown", "md", "text/markdown"),
        ("html", "html", "text/html"),
    ]
    for column, (fmt, extension, mime) in zip(st.columns(3), formats, strict=True):
        column.download_button(
            f"Download {fmt.upper() if fmt != 'markdown' else 'Markdown'}",
            render_report(report, fmt),
            file_name=f"{report.id}.{extension}",
            mime=mime,
        )


def main() -> None:
    """Open the same SQLite file as the gateway using a read-only connection."""
    st.set_page_config(page_title="Cerberus · Guard console", page_icon="🛡️", layout="wide")
    st.html(STYLE)
    st.html(
        '<div class="c-brand"><span class="c-mark" aria-hidden="true"><i></i><i></i><i></i></span>'
        "Cerberus<small>Guard console</small></div>"
    )
    nav, window, refresh = st.columns([3, 1, 1], vertical_alignment="center")
    view = nav.radio(
        "Console view", ["Activity", "Reports"], horizontal=True, label_visibility="collapsed"
    )
    live = False
    limit = 50
    if view == "Activity":
        limit = window.selectbox(
            "Request window",
            [50, 100, 250],
            format_func=lambda n: f"Last {n} requests",
            label_visibility="collapsed",
        )
        live = refresh.toggle("Live · 5s", value=True)
    database = os.getenv("CERBERUS_DATABASE_URL", "sqlite:///cerberus.sqlite3")

    @st.fragment(run_every=5 if live else None)
    def refresh_view() -> None:
        """Refresh from a fresh database snapshot while this browser session is open."""
        try:
            with closing(AuditStore(database, read_only=True)) as store:
                if view == "Activity":
                    show_activity(store, limit)
                else:
                    show_reports(store)
        except (SQLAlchemyError, ValueError):
            st.warning(
                "Audit data is unavailable. Start the gateway or demo and check CERBERUS_DATABASE_URL."
            )

    refresh_view()


def launch() -> None:
    """Launch the installed optional dashboard on loopback through Streamlit's CLI."""
    from streamlit.web import cli

    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    cli.main()


if __name__ == "__main__":
    main()
