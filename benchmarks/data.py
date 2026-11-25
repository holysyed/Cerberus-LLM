"""Load frozen labels and materialize public text from checksum-verified test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from cerberus.core.models import GuardContext

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT.parent / ".cache" / "phase6"
PER_CLASS = 24


class Example(BaseModel):
    """One labeled observation, with context passed through the real guard API."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    id: str
    task: Literal["input", "pii", "safety", "grounding", "schema"]
    positive: bool
    text: str
    source: str = "cerberus"
    context: GuardContext = Field(default_factory=GuardContext)


@dataclass(frozen=True)
class Split:
    """An immutable upstream test artifact."""

    name: str
    repository: str
    revision: str
    path: str
    sha256: str

    @property
    def url(self) -> str:
        """Public artifact at its pinned upstream revision."""
        return (
            f"https://huggingface.co/datasets/{self.repository}/resolve/{self.revision}/{self.path}"
        )


SPLITS = (
    Split(
        "deepset",
        "deepset/prompt-injections",
        "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "data/test-00000-of-00001-701d16158af87368.parquet",
        "39ac797cabc157eeed58435a08593b2952bb6cb16fc394a2d383f447cc7b246e",
    ),
    Split(
        "safeguard",
        "xTRam1/safe-guard-prompt-injection",
        "a3a877d608f37b7d20d9945671902df895ecdb46",
        "data/test-00000-of-00001.parquet",
        "99981a32fa354d9f541aea7214c15fed258c67639dcb3da5751968b043d8f742",
    ),
)


def digest(data: bytes) -> str:
    """Return a stable content identifier."""
    return hashlib.sha256(data).hexdigest()


def write_artifact(path: Path, content: bytes) -> None:
    """Publish a complete file so interrupted or concurrent runs never expose partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def synthetic_cases() -> list[Example]:
    """Read the authored cases and validate all labels, contexts and identifiers."""
    rows = (ROOT / "synthetic.v1.jsonl").read_text("utf-8").splitlines()
    cases = [Example.model_validate_json(row) for row in rows if row.strip()]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Duplicate synthetic case ID")
    return cases


def public_cases(cache: Path, *, download: bool = False) -> list[Example]:
    """Select a balanced panel by text hash before looking at any predictions."""
    pq = import_module("pyarrow.parquet")

    cases = []
    seen: set[str] = set()
    for split in SPLITS:
        path = cache / f"{split.name}.parquet"
        if download and (not path.exists() or digest(path.read_bytes()) != split.sha256):
            response = httpx.get(split.url, follow_redirects=True, timeout=60)
            response.raise_for_status()
            if digest(response.content) != split.sha256:
                raise ValueError(f"Download checksum mismatch for {split.name}")
            write_artifact(path, response.content)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python -m benchmarks.data --download")
        if digest(path.read_bytes()) != split.sha256:
            raise ValueError(f"Checksum mismatch for {split.name}")
        table = pq.read_table(path, columns=["text", "label"]).to_pylist()
        candidates: list[tuple[str, int, str, bool]] = []
        for i, row in enumerate(table):
            if row["label"] not in (0, 1) or not isinstance(row["text"], str):
                raise ValueError("Unexpected public dataset schema")
            canonical = " ".join(row["text"].casefold().split())
            if canonical:
                candidates.append((digest(canonical.encode()), i, row["text"], bool(row["label"])))
        counts = {False: 0, True: 0}
        for key, i, text, positive in sorted(candidates):
            if key in seen or counts[positive] >= PER_CLASS:
                continue
            seen.add(key)
            counts[positive] += 1
            cases.append(
                Example(
                    id=f"{split.name}-test-{i}",
                    task="input",
                    source=split.name,
                    positive=positive,
                    text=text,
                )
            )
        if min(counts.values()) != PER_CLASS:
            raise ValueError(f"Insufficient distinct examples in {split.name}")
    return cases


def panel_manifest(cases: list[Example]) -> list[dict[str, str | bool]]:
    """Identify public samples without redistributing the prompt strings."""
    return [
        {"id": c.id, "positive": c.positive, "text_sha256": digest(c.text.encode())} for c in cases
    ]


def load_cases(cache: Path, *, download: bool = False) -> list[Example]:
    """Reject drift in the public panel before running any detector."""
    public = public_cases(cache, download=download)
    frozen = json.loads((ROOT / "public.v1.json").read_text("utf-8"))
    if panel_manifest(public) != frozen:
        raise ValueError("Public sample selection differs from the frozen manifest")
    return public + synthetic_cases()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned test files")
    args = parser.parse_args()
    print(f"Verified {len(load_cases(args.cache, download=args.download))} benchmark cases.")
