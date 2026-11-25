"""Checks for the installed offline evaluation entry point."""

import json
from pathlib import Path

import pytest

from cerberus.redteam.__main__ import main


def test_report_bundle(tmp_path: Path) -> None:
    """Each run creates a new directory containing three consistent formats."""
    assert main(["--out", str(tmp_path)]) == 0
    (folder,) = tmp_path.iterdir()
    data = json.loads((folder / "report.json").read_text("utf-8"))
    assert data["id"] == folder.name
    for name in ("report.md", "report.html"):
        assert (folder / name).stat().st_size > 100


@pytest.mark.parametrize("data", [b"[]", b"{", b"x" * 262_145])
def test_invalid_recordings(data: bytes, tmp_path: Path) -> None:
    """Reject malformed and oversized input files before evaluation."""
    path = tmp_path / "recordings.json"
    path.write_bytes(data)
    with pytest.raises(SystemExit):
        main(["--recordings", str(path), "--out", str(tmp_path)])
