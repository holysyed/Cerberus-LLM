"""Measure Cerberus decisions on frozen labels. Run from the repository root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import httpx

from benchmarks.data import DEFAULT_CACHE, ROOT, Example, digest, load_cases, write_artifact
from benchmarks.metrics import Observation, percentile, summarize
from cerberus.core import Guard, GuardContext, GuardResult, Pipeline, Policy, load_policy
from cerberus.guards.detectors import DEFAULT_MODEL, DEFAULT_REVISION, HFClassifier, LLMJudge
from cerberus.guards.factory import Guardrails
from cerberus.guards.injection import InputGuard
from cerberus.guards.pii import PiiGuard
from cerberus.guards.safety import GroundingGuard, SafetyGuard
from cerberus.guards.schema import SchemaGuard
from cerberus.providers.client import ProviderConfig, build_provider

SEED = 615
WARMUP = "When does the garden gate open?"
JUDGE_ORIGIN = "http://127.0.0.1:11434"


def elapsed(start: int) -> float:
    """Convert monotonic nanoseconds to milliseconds."""
    return (perf_counter_ns() - start) / 1_000_000


@dataclass
class TimedLayer:
    """Record backend calls inside a real stacked guard without repeating inference."""

    detect: Callable[[str], float]
    scores: list[float] = field(default_factory=list)
    elapsed_ms: float = 0.0
    unavailable: bool = False

    def reset(self) -> None:
        """Start a fresh per-source sample group for the next request."""
        self.scores.clear()
        self.elapsed_ms = 0.0
        self.unavailable = False

    def __call__(self, text: str) -> float:
        """Preserve the detector's value or exception and measure its runtime."""
        start = perf_counter_ns()
        try:
            value = self.detect(text)
            if not 0 <= value <= 1:
                raise ValueError("Invalid detector score")
            self.scores.append(value)
            return value
        except Exception:
            self.unavailable = True
            raise
        finally:
            self.elapsed_ms += elapsed(start)

    def decision(self, threshold: float) -> bool | None:
        """A raw layer fires if any source reaches its configured confidence gate."""
        if self.unavailable or not self.scores:
            return None
        return max(self.scores) >= threshold


def prediction(decisions: tuple[GuardResult, ...]) -> bool | None:
    """A positive means a finding, including PII that was successfully sanitized."""
    for result in decisions:
        if result.category == "availability" or any(
            reason.endswith("_unavailable") for reason in result.reasons
        ):
            return None
    return any(not result.passed for result in decisions)


class Measurements:
    """Accumulate outcomes and durations without retaining input text or responses."""

    def __init__(self) -> None:
        """Initialize independent profiles."""
        self.rows: dict[str, list[Observation]] = {}
        self.guard_path_ms: dict[str, list[float]] = {}

    def add(self, profile: str, case: Example, verdict: bool | None, ms: float) -> None:
        """Record exactly one observation for this profile and case."""
        self.rows.setdefault(profile, []).append(
            Observation(case.positive, verdict, ms, case.source)
        )

    def report(self) -> dict[str, Any]:
        """Aggregate each profile and expose stratified counts for interpretation."""
        profiles = {}
        for name, rows in self.rows.items():
            groups = sorted({row.source for row in rows})
            profiles[name] = {
                "aggregate": summarize(rows),
                "by_source": {
                    source: summarize([row for row in rows if row.source == source])
                    for source in groups
                },
            }
        return {
            "profiles": profiles,
            "admitted_guard_path": {
                name: {
                    "cases": len(ms),
                    "p50_ms": percentile(ms, 0.5),
                    "p95_ms": percentile(ms, 0.95),
                }
                for name, ms in self.guard_path_ms.items()
            },
        }


def input_profile(
    name: str,
    policy: Policy,
    cases: list[Example],
    results: Measurements,
    *,
    classifier: TimedLayer | None = None,
    judge: TimedLayer | None = None,
) -> None:
    """Exercise the real input pipeline at the strict operating point for each ablation."""
    settings = policy.input.model_copy(
        update={"ml_enabled": classifier is not None, "judge_enabled": judge is not None}
    )
    active = policy.model_copy(update={"input": settings})
    pipeline = Pipeline([InputGuard(active, classifier, judge)], active)
    output = Guardrails.build(policy).outputs
    layers = [layer for layer in (classifier, judge) if layer is not None]
    warm = pipeline.run(WARMUP)
    if prediction(warm.results) is None:
        raise RuntimeError(f"{name} failed its detector warm-up")
    output.run("The garden opens in the morning.")
    admitted: list[float] = []
    results.guard_path_ms[name] = admitted
    for case in cases:
        for layer in layers:
            layer.reset()
        start = perf_counter_ns()
        result = pipeline.run(case.text, case.context)
        duration = elapsed(start)
        decision = prediction(result.results)
        if decision is not None and not result.blocked:
            start_output = perf_counter_ns()
            released = output.run("The garden opens in the morning.", case.context)
            if released.blocked:
                raise RuntimeError(
                    "The fixed benign response unexpectedly failed the output pipeline"
                )
            admitted.append(duration + elapsed(start_output))
        results.add(name, case, decision, duration)
        # Observe raw backends in their first measured profile, once per frozen case.
        if classifier is not None and judge is None:
            results.add(
                "Classifier alone",
                case,
                classifier.decision(settings.ml_threshold),
                classifier.elapsed_ms,
            )
        if judge is not None:
            results.add(
                "Judge alone", case, judge.decision(settings.judge_threshold), judge.elapsed_ms
            )


def output_profiles(policy: Policy, cases: list[Example], results: Measurements, ner: bool) -> None:
    """Evaluate each output detector on relevant labels and the strict stack on its tasks."""
    evidence = policy.model_copy(
        update={"output": policy.output.model_copy(update={"grounding_enabled": True})}
    )
    checks: list[tuple[str, str, Guard]] = [
        ("PII", "pii", PiiGuard(policy)),
        ("Safety", "safety", SafetyGuard(policy)),
        ("Grounding", "grounding", GroundingGuard(evidence)),
        ("Schema", "schema", SchemaGuard(policy)),
    ]
    if ner:
        nlp = policy.model_copy(
            update={"output": policy.output.model_copy(update={"ner_enabled": True})}
        )
        checks.append(("PII + spaCy", "pii", PiiGuard(nlp)))
    for name, task, guard in checks:
        warm_context = (
            next((case.context for case in cases if case.task == "schema"), GuardContext())
            if task == "schema"
            else GuardContext()
        )
        if prediction((guard.check(WARMUP, warm_context),)) is None:
            raise RuntimeError(f"{name} failed its warm-up")
        for case in cases:
            if case.task == task:
                start = perf_counter_ns()
                finding = guard.check(case.text, case.context)
                results.add(name, case, prediction((finding,)), elapsed(start))
    output = Guardrails.build(policy).outputs
    for case in cases:
        if case.task in ("pii", "safety", "schema"):
            start = perf_counter_ns()
            result = output.run(case.text, case.context)
            results.add("Strict output pipeline", case, prediction(result.results), elapsed(start))


def heuristic_size_profile(policy: Policy) -> dict[str, Any]:
    """Measure the heuristic pipeline on both a full source and a full RAG request."""
    pipeline = Pipeline([InputGuard(policy)], policy)
    line = "The weekday library timetable lists the opening hours for each reading room. "
    source = (line * (policy.input.max_source_chars // len(line) + 1))[
        : policy.input.max_source_chars
    ]
    cases = {
        "one_source": (source, GuardContext()),
        "two_sources": (source, GuardContext(documents=(source,))),
    }
    result = {}
    for name, (text, context) in cases.items():
        timings = []
        for i in range(41):
            start = perf_counter_ns()
            decision = pipeline.run(text, context)
            duration = elapsed(start)
            if decision.blocked:
                raise RuntimeError("Long benign input was blocked during timing")
            if i:
                timings.append(duration)
        result[name] = {
            "characters": len(text) + sum(map(len, context.documents)),
            "repeats": len(timings),
            "p50_ms": percentile(timings, 0.5),
            "p95_ms": percentile(timings, 0.95),
        }
    return result


def table(report: dict[str, Any]) -> str:
    """Render the published table directly from the stored measurements."""
    lines = [
        "| Profile | Scored / cases | Precision | Recall | F1 | FPR | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, profile in report["profiles"].items():
        m = profile["aggregate"]
        cells = [
            "—" if m[k] is None else f"{m[k]:.3f}"
            for k in ("precision", "recall", "f1", "fpr", "p50_ms", "p95_ms")
        ]
        lines.append(" | ".join(["| " + name, f"{m['scored']} / {m['cases']}", *cells]) + " |")
    return "\n".join(lines)


def main() -> None:
    """Benchmark the installed package and write one machine-readable result artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--full", action="store_true", help="Include real classifier and local Ollama judge"
    )
    parser.add_argument(
        "--classifier", default=DEFAULT_MODEL, help="Pinned FMOps model or a local snapshot"
    )
    parser.add_argument("--judge-model", default="qwen2.5:0.5b")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--ner", action="store_true", help="Also measure installed spaCy PERSON detector"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE / "results.json")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    cases = load_cases(args.cache)
    random.Random(SEED).shuffle(cases)
    inputs = [case for case in cases if case.task == "input"]
    outputs = [case for case in cases if case.task != "input"]
    policy = load_policy()
    results = Measurements()
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine(),
    }
    print(f"Guard baselines on {len(inputs)} input and {len(outputs)} output cases", flush=True)
    input_profile("Heuristics", policy, inputs, results)
    if args.ner:
        env["ner"] = {
            "spacy": version("spacy"),
            "model": "en_core_web_sm",
            "version": version("en-core-web-sm"),
        }
    if args.full:
        torch = import_module("torch")
        torch.set_num_threads(args.threads)
        classifier = TimedLayer(HFClassifier(args.classifier))
        env["classifier"] = {
            "model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION if args.classifier == DEFAULT_MODEL else None,
            "device": "cpu",
            "torch": version("torch"),
            "threads": args.threads,
            "transformers": version("transformers"),
        }
        if Path(args.classifier).is_dir():
            fingerprints = {}
            for filename in (
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
            ):
                with (Path(args.classifier) / filename).open("rb") as stream:
                    fingerprints[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
            env["classifier"]["local_sha256"] = fingerprints
        print("Measuring heuristics and DistilBERT", flush=True)
        input_profile("Heuristics + ML", policy, inputs, results, classifier=classifier)
        with httpx.Client(base_url=JUDGE_ORIGIN, trust_env=False, timeout=15) as local:
            tags = local.get("/api/tags").raise_for_status().json()["models"]
            tag = next((item for item in tags if item["name"] == args.judge_model), None)
            if tag is None:
                raise ValueError(f"Local judge model is missing: {args.judge_model}")
            env["judge"] = {
                "model": args.judge_model,
                "digest": tag["digest"],
                "ollama": local.get("/api/version").raise_for_status().json()["version"],
                "temperature": 0,
                "seed": 42,
                "max_tokens": 160,
            }
        provider = build_provider(
            ProviderConfig(
                kind="ollama", base_url=JUDGE_ORIGIN, model=args.judge_model, max_tokens=160
            )
        )
        print("Measuring heuristics, DistilBERT and the local Ollama judge", flush=True)
        try:
            input_profile(
                "Heuristics + ML + judge",
                policy,
                inputs,
                results,
                classifier=classifier,
                judge=TimedLayer(LLMJudge(provider.judge_complete)),
            )
        finally:
            provider.close()
    output_profiles(policy, outputs, results, args.ner)
    report = {
        "suite": "cerberus-evaluation-v1",
        "utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "dataset_sha256": {
            name: digest((ROOT / name).read_text("utf-8").encode())
            for name in ("public.v1.json", "synthetic.v1.jsonl")
        },
        "strict_policy": policy.model_dump(),
        "environment": env,
        "heuristic_size_profile": heuristic_size_profile(policy),
        **results.report(),
    }
    write_artifact(args.output, (json.dumps(report, indent=2) + "\n").encode())
    print(table(report))


if __name__ == "__main__":
    main()
