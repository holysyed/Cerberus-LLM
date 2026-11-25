"""PII and secret filtering for prose and structured JSON outputs."""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from functools import partial
from threading import Lock
from time import perf_counter
from typing import Any, NamedTuple

from cerberus.core.models import Guard, GuardContext, GuardResult
from cerberus.core.policy import Policy


class _Span(NamedTuple):
    """A half-open sensitive range and its display label."""

    start: int
    end: int
    kind: str


_EMAIL = re.compile(
    r"(?i)(?<![\w.+%-])[a-z0-9._%+-]{1,64}@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,63}\b"
)
_AADHAAR = re.compile(
    r"(?<![\w+])(?<![0-9][ -])[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}(?!\w|[ -][0-9]{4}\b)"
)
_PAN = re.compile(r"(?i)\b[a-z]{5}[0-9]{4}[a-z]\b")
_IFSC = re.compile(r"(?i)\b[a-z]{4}0[a-z0-9]{6}\b")
_PHONE = re.compile(
    r"(?<!\w)(?:\+91[ -]?[6-9](?:[ -]?[0-9]){9}|[6-9][0-9]{9}"
    r"|(?:\+1[ .-]?)?\(?[2-9][0-9]{2}\)?[ .-][0-9]{3}[ .-][0-9]{4}"
    r"|\+[1-9][0-9]{7,14})(?!\w)"
)
_SSN = re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b")
_CARD = re.compile(r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![0-9])")
_PREFIX = re.compile(
    r"(?<!\w)(?:(?:sk-|sk_live_|rk_live_|gh[pousr]_|github_pat_|xox[baprs]-|AIza)"
    r"[a-zA-Z0-9_+/.=-]+|(?:AKIA|ASIA)[A-Z0-9]{16})"
)
_PEM = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"
    r".*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)
_SECRET_WORDS = r"(?:api[ _-]?key|access[ _-]?token|client[ _-]?secret|password|secret)"
_LABELED_SECRET = re.compile(
    rf"""\b{_SECRET_WORDS}["']?\s*[:=]\s*(?:"([^"\n]+)"|'([^'\n]+)'|([^\s,;{{}}\[\]"']+))""",
    re.IGNORECASE,
)
_NAME = re.compile(
    r"(?i:\b(?:my name is|full name\s*[:=]|name\s*[:=]))[ \t]+"
    r"([A-Z][^\W\d_]*(?:[-'][^\W\d_]+)*(?:[ \t]+[A-Z][^\W\d_]*(?:[-'][^\W\d_]+)*){0,4})"
)
_TOKEN = re.compile(r"(?<![a-zA-Z0-9_+/=-])[a-zA-Z0-9_+/=-]{24,}")
_STRUCTURED_KEYS = {
    "name": "name",
    "full_name": "name",
    "first_name": "name",
    "last_name": "name",
    "api_key": "secret",
    "apikey": "secret",
    "access_token": "secret",
    "token": "secret",
    "password": "secret",
    "client_secret": "secret",
    "secret": "secret",
}
_ORDER = ("secret", "card", "aadhaar", "ssn", "email", "phone", "pan", "ifsc", "name")
_PATTERNS = (
    (_EMAIL, "email"),
    (_AADHAAR, "aadhaar"),
    (_PAN, "pan"),
    (_IFSC, "ifsc"),
    (_PHONE, "phone"),
    (_SSN, "ssn"),
    (_PREFIX, "secret"),
    (_PEM, "secret"),
)
_PLACEHOLDER = re.compile(
    r"\[(?:NAME|EMAIL|PHONE|CARD|AADHAAR|PAN|IFSC|SECRET|SSN)"
    r"(?::(?:[0-9a-f]{24}|\*{4}[0-9]{0,4}))?\]"
)


def _luhn(value: str) -> bool:
    """Check a payment-card candidate without retaining its digits."""
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) < 2:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _high_entropy(value: str) -> bool:
    """Require diverse characters and a token-like mix before scoring entropy."""
    mixed_case = not value.islower() and not value.isupper()
    if not any(c.isalpha() for c in value) or not (mixed_case or any(c.isdigit() for c in value)):
        return False
    counts = Counter(value)
    return -sum((n / len(value)) * math.log2(n / len(value)) for n in counts.values()) >= 4.25


def _find_spans(text: str) -> list[_Span]:
    """Collect candidate ranges without putting their content into metadata."""
    spans = [_Span(*m.span(), kind) for regex, kind in _PATTERNS for m in regex.finditer(text)]
    spans.extend(_Span(*m.span(), "card") for m in _CARD.finditer(text) if _luhn(m.group()))
    spans.extend(_Span(*m.span(1), "name") for m in _NAME.finditer(text))
    for m in _LABELED_SECRET.finditer(text):
        group = m.lastindex
        if group is not None and not _PLACEHOLDER.fullmatch(m[group]):
            spans.append(_Span(*m.span(group), "secret"))
    spans.extend(_Span(*m.span(), "secret") for m in _TOKEN.finditer(text) if _high_entropy(m[0]))
    return spans


def _merge(spans: list[_Span]) -> list[_Span]:
    """Cover the union of overlapping ranges, retaining the most restrictive label."""
    result: list[_Span] = []
    for span in sorted(spans):
        if result and span.start < result[-1].end:
            previous = result.pop()
            kind = min((previous.kind, span.kind), key=_ORDER.index)
            result.append(_Span(previous.start, max(previous.end, span.end), kind))
        else:
            result.append(span)
    return result


class _AmbiguousStructure(ValueError):
    """A structure cannot be transformed without dropping field data."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys before applying any security decision."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise _AmbiguousStructure
    return result


class PiiGuard(Guard):
    """Redact, mask, or pseudonymize sensitive output before release."""

    name = "pii"

    def __init__(self, policy: Policy, *, hash_key: bytes | None = None) -> None:
        """Use a deployment-supplied, at least 256-bit HMAC key in hash mode."""
        self.policy = policy
        self._hash_key = hash_key
        if policy.output.pii_mode == "hash" and (hash_key is None or len(hash_key) < 32):
            raise ValueError("PII hash mode requires a key of at least 32 bytes")
        self._nlp: Any = None
        self._load_lock = Lock()

    def _persons(self, text: str) -> list[_Span]:
        """Load the administrator-selected spaCy model on its first use."""
        if self._nlp is None:
            with self._load_lock:
                if self._nlp is None:
                    self._nlp = importlib.import_module("spacy").load(self.policy.output.ner_model)
        return [
            _Span(ent.start_char, ent.end_char, "name")
            for ent in self._nlp(text).ents
            if ent.label_ in ("PERSON", "PER")
        ]

    def _replace(self, kind: str, value: str) -> str:
        """Build a replacement that discloses only policy-authorized information."""
        label = kind.upper()
        mode = self.policy.output.pii_mode
        if mode == "hash":
            assert self._hash_key is not None
            message = kind.encode() + b"\x00" + value.encode("utf-8", errors="surrogatepass")
            digest = hmac.new(self._hash_key, message, hashlib.sha256).hexdigest()[:24]
            return f"[{label}:{digest}]"
        if mode == "mask":
            suffix = re.sub(r"\D", "", value)[-4:] if kind in ("card", "phone") else ""
            return f"[{label}:****{suffix}]"
        return f"[{label}]"

    def _filter_text(self, text: str, kind: str | None, found: set[str]) -> str:
        """Filter one decoded string, optionally treating the entire field as sensitive."""
        if not text or _PLACEHOLDER.fullmatch(text):
            return text
        if kind:
            found.add(kind)
            return self._replace(kind, text)
        spans = _find_spans(text)
        if self.policy.output.ner_enabled and "ner_unavailable" not in found:
            try:
                spans.extend(self._persons(text))
            except Exception:  # noqa: BLE001 - third-party inference errors obey the configured policy
                found.add("ner_unavailable")
        chunks: list[str] = []
        pos = 0
        for start, end, label in _merge(spans):
            found.add(label)
            chunks.extend((text[pos:start], self._replace(label, text[start:end])))
            pos = end
        return "".join(chunks) + text[pos:]

    def check(self, text: str, context: GuardContext) -> GuardResult:
        """Inspect raw prose or decoded JSON, returning only payload-free reason codes."""
        started = perf_counter()
        found: set[str] = set()
        replacement: str | None = None
        score = 0.0
        if len(text) > self.policy.output.max_chars:
            found.add("output_length_limit")
            score = 1.0
        else:
            clean = partial(self._filter_text, found=found)
            try:
                if text.lstrip().startswith(("{", "[", '"')):
                    try:
                        obj = json.loads(text, object_pairs_hook=_unique_object)
                    except json.JSONDecodeError:
                        replacement = clean(text, None)
                    else:
                        obj = _filter_json(obj, None, clean)
                        replacement = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
                else:
                    replacement = clean(text, None)
                if found - {"ner_unavailable"}:
                    score = 0.85
                if "ner_unavailable" in found and self.policy.fail_mode == "closed":
                    score = 1.0
                    replacement = None
            except (_AmbiguousStructure, RecursionError):
                found.add("ambiguous_json")
                score, replacement = 1.0, None
        threshold = self.policy.for_guard(self.name).threshold
        return GuardResult(
            passed=score < (self.policy.threshold if threshold is None else threshold),
            risk_score=score,
            category=self.name,
            reasons=tuple(sorted(found)),
            sanitized_text=replacement if score else None,
            latency_ms=(perf_counter() - started) * 1000,
        )


def _filter_json(obj: Any, kind: str | None, clean: Callable[[str, str | None], str]) -> Any:
    """Inspect decoded property names and values, preserving all distinct fields."""
    if isinstance(obj, dict):
        fields: dict[str, Any] = {}
        for key, value in obj.items():
            safe_key = clean(key, None)
            if safe_key in fields:
                raise _AmbiguousStructure
            field_kind = _STRUCTURED_KEYS.get(key.casefold().replace("-", "_"), kind)
            fields[safe_key] = _filter_json(value, field_kind, clean)
        return fields
    if isinstance(obj, list):
        return [_filter_json(item, kind, clean) for item in obj]
    if isinstance(obj, str):
        return clean(obj, kind)
    if isinstance(obj, (float, int)) and not isinstance(obj, bool):
        original = str(obj)
        result = clean(original, kind)
        return result if result != original else obj
    return obj
