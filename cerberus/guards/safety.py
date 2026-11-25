"""Bounded output safety checks and optional lexical evidence checks."""

import json
import re
import unicodedata
from time import perf_counter

from cerberus.core.models import Guard, GuardContext, GuardResult
from cerberus.core.policy import Policy

_WORDS = re.compile(r"[^\W_]+")
_TOXIC = re.compile(r"\byou(?:'re| are)(?: (?:a|an|such|complete|utter))* (?:idiot|moron|scum)\b")
_THREAT = re.compile(r"\bi(?:'ll| will) (?:seriously |physically )?(?:hurt|kill|assault) you\b")
_UNSAFE = re.compile(r"\byou (?:should|ought to) (?:hurt|kill) yourself\b")
_CONTRADICTORY_REFUSAL = re.compile(
    r"\bi (?:can(?:not|'t)|won't|will not) (?:comply|assist|help|provide|share|reveal)\b"
    r".{0,180}?\b(?:but|however|nevertheless)\b.{0,64}?"
    r"\b(?:here (?:are|is)|i(?:'ll| will) (?:provide|share|reveal)) "
    r"(?:the |your )?(?:restricted|forbidden|private|secret|hidden)\b",
    re.DOTALL,
)
_CLAIM = re.compile(r"[^.!?\n]+[.!?]?")
_STOP_WORD_TEXT = (
    "a an the this that these those is am are was were been be being of on at as by for from "
    "and or nor with to in it its has have had will can i we they you"
)
_STOP_WORDS = frozenset(_STOP_WORD_TEXT.split())


def _fold(text: str) -> str:
    """Normalize punctuation and case without retaining a copy outside this call."""
    return unicodedata.normalize("NFKC", text).casefold().replace("’", "'")


def _inspection_text(text: str) -> str:
    """Include decoded strings in escaped JSON, traversing containers iteratively."""
    if "\\" not in text or not text.lstrip().startswith(("{", "[", '"')):
        return _fold(text)
    try:
        pending: list[object] = [json.loads(text)]
    except (ValueError, RecursionError):
        return _fold(text)
    parts = [text]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return _fold("\n".join(parts))


def _leaks_prompt(output: str, system: str) -> bool:
    """Identify literal prompt disclosure or an eight-token contiguous excerpt."""
    prompt_words = _WORDS.findall(_fold(system))
    if len(prompt_words) < 8:
        phrase = " ".join(prompt_words)
        return len(phrase) >= 12 and f" {phrase} " in f" {' '.join(_WORDS.findall(output))} "
    output_words = _WORDS.findall(output)
    windows = {tuple(prompt_words[i : i + 8]) for i in range(len(prompt_words) - 7)}
    return any(tuple(output_words[i : i + 8]) in windows for i in range(len(output_words) - 7))


class _OutputCheck(Guard):
    """Shared timing and configured threshold handling for these output checks."""

    def __init__(self, policy: Policy) -> None:
        """Retain an administrator's validated policy."""
        self.policy = policy

    def _result(
        self, score: float, category: str, reasons: tuple[str, ...], started: float
    ) -> GuardResult:
        """Build a content-free finding from local evidence."""
        limit = self.policy.for_guard(self.name).threshold
        return GuardResult(
            passed=score < (self.policy.threshold if limit is None else limit),
            risk_score=score,
            category=category,
            reasons=reasons,
            latency_ms=(perf_counter() - started) * 1000.0,
        )


class SafetyGuard(_OutputCheck):
    """Flag targeted abuse, direct threats, contradictory refusals, and prompt disclosure."""

    name = "safety"

    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Evaluate the generated response with optional trusted system-prompt metadata."""
        started = perf_counter()
        limit = self.policy.output.max_chars
        if len(text) > limit or len(context.system_prompt or "") > limit:
            return self._result(1.0, "output_limit", ("length_limit",), started)
        output = _inspection_text(text)
        if context.system_prompt and _leaks_prompt(output, context.system_prompt):
            return self._result(1.0, "prompt_leakage", ("system_prompt_overlap",), started)
        if _THREAT.search(output) or _UNSAFE.search(output):
            return self._result(0.98, "unsafe_content", ("direct_harm_language",), started)
        if _CONTRADICTORY_REFUSAL.search(output):
            return self._result(0.9, "refusal_bypass", ("contradictory_refusal",), started)
        if _TOXIC.search(output):
            return self._result(0.85, "toxicity", ("targeted_abuse",), started)
        return self._result(0.0, "safety", (), started)


class GroundingGuard(_OutputCheck):
    """Compare claim terms and numbers with the supplied retrieval and tool evidence."""

    name = "grounding"

    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Flag low lexical coverage or a new number within a declarative sentence."""
        started = perf_counter()
        settings = self.policy.output
        if not settings.grounding_enabled:
            return self._result(0.0, "grounding", (), started)
        input_limits = self.policy.input
        if (
            len(text) > settings.max_chars
            or 1 + len(context.documents) + len(context.tool_outputs) > input_limits.max_sources
        ):
            return self._result(1.0, "grounding", ("grounding_limit",), started)
        evidence = context.documents + context.tool_outputs
        if sum(map(len, evidence)) > input_limits.max_total_chars:
            return self._result(1.0, "grounding", ("grounding_limit",), started)
        known = {word for source in evidence for word in _WORDS.findall(_fold(source))}
        for sentence in _CLAIM.finditer(_fold(text)):
            if sentence.group().endswith("?"):
                continue
            words = set(_WORDS.findall(sentence.group())) - _STOP_WORDS
            if not words:
                continue
            novel = words - known
            if any(word.isnumeric() for word in novel):
                return self._result(0.85, "grounding", ("unsupported_number",), started)
            coverage = 1.0 - len(novel) / len(words)
            if coverage < settings.grounding_threshold:
                return self._result(0.8, "grounding", ("insufficient_term_support",), started)
        return self._result(0.0, "grounding", (), started)
