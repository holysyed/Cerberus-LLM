"""Regression checks for benchmark labels, denominators, timing and report provenance."""

import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks import data
from benchmarks.data import ROOT, Example
from benchmarks.metrics import Observation, percentile, summarize
from benchmarks.run import Measurements, TimedLayer, input_profile, output_profiles, table
from cerberus.core import GuardContext, load_policy


def test_published_results_match_frozen_data_and_readme() -> None:
    """Catch drift between the reference artifact, its labels, and the published table."""
    report = json.loads((ROOT / "results.v1.json").read_text("utf-8"))
    for name, expected in report["dataset_sha256"].items():
        assert data.digest((ROOT / name).read_text("utf-8").encode()) == expected
    assert table(report) in (ROOT.parent / "README.md").read_text("utf-8")
    for profile in report["profiles"].values():
        aggregate = profile["aggregate"]
        assert sum(aggregate[k] for k in ("tp", "fp", "fn", "tn")) == aggregate["scored"]
        assert aggregate["scored"] + aggregate["unavailable"] == aggregate["cases"]
        for k in ("tp", "fp", "fn", "tn", "unavailable", "cases"):
            assert sum(m[k] for m in profile["by_source"].values()) == aggregate[k]
    for key in ("Heuristics", "Heuristics + ML", "Heuristics + ML + judge"):
        assert report["profiles"][key]["aggregate"]["cases"] == 128


def test_metrics_exclude_unavailable_from_accuracy_and_latency() -> None:
    """A detector failure cannot receive credit for blocking a positive case."""
    outcomes = [(True, True), (False, True), (True, False), (False, False), (True, None)]
    rows = [
        Observation(actual, predicted, i + 1.0, "test")
        for i, (actual, predicted) in enumerate(outcomes)
    ]
    metrics = summarize(rows)
    for key in ("tp", "fp", "fn", "tn", "unavailable"):
        assert metrics[key] == 1
    for key in ("precision", "recall", "f1", "fpr"):
        assert metrics[key] == 0.5
    assert metrics["p50_ms"] == 2.5
    assert metrics["p95_ms"] == pytest.approx(3.85)
    assert summarize([])["f1"] is None
    assert summarize([Observation(True, False, 1.0, "test")])["precision"] is None
    assert percentile([1], 0.95) == 1


def test_public_and_authored_labels_are_frozen_and_distinct() -> None:
    """A stable panel gives every required task both positive and negative cases."""
    public = json.loads((ROOT / "public.v1.json").read_text("utf-8"))
    assert len(public) == len({row["text_sha256"] for row in public}) == 96
    assert set(Counter((row["id"].split("-")[0], row["positive"]) for row in public).values()) == {
        24
    }
    cases = data.synthetic_cases()
    balance = Counter((c.task, c.positive) for c in cases)
    assert len(balance) == 10
    assert all(balance[task, True] == balance[task, False] for task, _ in balance)
    assert len({row["id"] for row in public} | {c.id for c in cases}) == 196


def test_public_cache_hash_selection_and_label_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic Parquet fixtures exercise deterministic sampling without network access."""
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "test.parquet"
    parquet.write_table(pa.table({"text": [" a ", "A", "b", "c"], "label": [0, 0, 1, 1]}), path)
    split = data.Split(
        "test", "example/fixture", "pinned", "test.parquet", data.digest(path.read_bytes())
    )
    monkeypatch.setattr(data, "SPLITS", (split,))
    monkeypatch.setattr(data, "PER_CLASS", 1)
    cases = data.public_cases(tmp_path)
    assert sorted(c.positive for c in cases) == [False, True]
    assert data.panel_manifest(cases) == data.panel_manifest(data.public_cases(tmp_path))
    with pytest.raises(ValueError, match="frozen manifest"):
        data.load_cases(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.public_cases(tmp_path)


def test_stacked_profiles_observe_all_sources_and_count_outages() -> None:
    """Raw detector scores are measured inside the same request as fused decisions."""

    def classify(text: str) -> float:
        """A scripted backend marks one document and fails on one tool result."""
        if text == "offline":
            raise ValueError("backend unavailable")
        return 0.8 if text == "toy marker" else 0.0

    cases = [
        Example(
            id="a",
            task="input",
            text="Summarize.",
            positive=True,
            context=GuardContext(documents=("toy marker",)),
        ),
        Example(
            id="b",
            task="input",
            text="Summarize.",
            positive=False,
            context=GuardContext(tool_outputs=("offline",)),
        ),
        Example(id="c", task="input", text="Read the timetable.", positive=False),
    ]
    results = Measurements()
    input_profile("stack", load_policy(), cases, results, classifier=TimedLayer(classify))
    report = results.report()
    for key in ("stack", "Classifier alone"):
        m = report["profiles"][key]["aggregate"]
        assert (m["tp"], m["tn"], m["unavailable"]) == (1, 1, 1)
    assert report["admitted_guard_path"]["stack"]["cases"] == 1


def test_output_measurement_counts_successful_sanitization() -> None:
    """A detector finding remains positive when the pipeline releases a replacement."""
    results = Measurements()
    cases = [Example(id="mail", task="pii", text="unit@example.invalid", positive=True)]
    output_profiles(load_policy(), cases, results, ner=False)
    report = results.report()
    assert report["profiles"]["Strict output pipeline"]["aggregate"]["tp"] == 1
    assert "Strict output pipeline" in table(report)
    assert "—" in table(report)
