"""Run the fixed offline evaluation and write all three report formats."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cerberus.core.policy import load_policy
from cerberus.redteam.evaluator import evaluate
from cerberus.redteam.export import render_report


def main(argv: Sequence[str] | None = None) -> int:
    """Write a new report bundle, optionally scoring supplied response recordings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="strict", help="Packaged strict profile or YAML file")
    parser.add_argument("--recordings", type=Path, help="JSON object mapping case IDs to responses")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="Parent output directory")
    args = parser.parse_args(argv)
    recordings: dict[str, str] | None = None
    if args.recordings:
        with args.recordings.open("rb") as stream:
            data = stream.read(262_145)
        if len(data) > 262_144:
            parser.error("recordings file exceeds 256 KiB")
        try:
            recordings = json.loads(data)
        except (UnicodeDecodeError, ValueError):
            parser.error("recordings must be a JSON object")
        if not isinstance(recordings, dict) or not all(
            isinstance(v, str) for v in recordings.values()
        ):
            parser.error("recordings must map case IDs to strings")
    report = evaluate(load_policy(args.policy), recordings)
    directory = args.out / report.id
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(render_report(report, "json"), encoding="utf-8")
    (directory / "report.md").write_text(render_report(report, "markdown"), encoding="utf-8")
    (directory / "report.html").write_text(render_report(report, "html"), encoding="utf-8")
    print(directory.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
