"""PII coverage using dummy identifiers, reserved addresses, and inert tokens."""

import json
from types import SimpleNamespace

import pytest

from cerberus.core.models import GuardContext
from cerberus.core.policy import OutputSettings, Policy
from cerberus.guards.pii import PiiGuard


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("me@example.test", "email"),
        ("+91 90000 00000", "phone"),
        ("(202) 555-0101", "phone"),
        ("+441632960123", "phone"),
        ("4242 4242 4242 4242", "card"),
        ("2345 0000 0000", "aadhaar"),
        ("TESTP0000X", "pan"),
        ("TEST0000001", "ifsc"),
        ("000-00-0000", "ssn"),
        ("sk-" + "fixture-value", "secret"),
        ("ABCDEabcdeFGHIJfghij0123456789", "secret"),
    ],
)
def test_filter_types(text: str, kind: str) -> None:
    """Every structured detector replaces the complete span with an anonymous token."""
    result = PiiGuard(Policy()).check(text, GuardContext())
    assert result.reasons == (kind,)
    assert result.sanitized_text == f"[{kind.upper()}]"
    assert not result.passed and result.latency_ms >= 0


@pytest.mark.parametrize(
    "text",
    [
        "A meeting in Spring Park",
        "order 12345",
        "4242424242424241",
        "4242 4242 4242 4241",
        "a" * 80,
        "Your name is on file.",
        "normal@example",
        "[]",
        '{"count":0,"ok":true,"missing":null}',
    ],
)
def test_clean_text(text: str) -> None:
    """Benign text and wrong card checksums are left alone."""
    result = PiiGuard(Policy()).check(text, GuardContext())
    assert result.passed and result.sanitized_text is None


def test_mask_and_hash() -> None:
    """Masks withhold short credentials and keyed digests are deployment-specific."""
    mask = PiiGuard(Policy(output=OutputSettings(pii_mode="mask")))
    assert mask.check("password='a'", GuardContext()).sanitized_text == "password='[SECRET:****]'"
    assert mask.check("4242424242424242", GuardContext()).sanitized_text == "[CARD:****4242]"
    assert mask.check("000-00-0000", GuardContext()).sanitized_text == "[SSN:****]"
    policy = Policy(output=OutputSettings(pii_mode="hash"))
    with pytest.raises(ValueError, match="32 bytes"):
        PiiGuard(policy, hash_key=b"weak")
    first = PiiGuard(policy, hash_key=b"a" * 32)
    second = PiiGuard(policy, hash_key=b"b" * 32)
    sample = "synthetic@example.test"
    token = first.check(sample, GuardContext()).sanitized_text
    assert token and token.startswith("[EMAIL:") and sample not in token
    assert token == first.check(sample, GuardContext()).sanitized_text
    assert token != second.check(sample, GuardContext()).sanitized_text


def test_decoded_keys_values_and_numeric_json() -> None:
    """Unicode escapes and JSON numeric typing do not suppress inspection."""
    text = (
        r'{"name":"Kai Example","user\u0040example.test":["a\u0040example.test",4242424242424242]}'
    )
    result = PiiGuard(Policy()).check(text, GuardContext())
    assert json.loads(result.sanitized_text or "") == {
        "name": "[NAME]",
        "[EMAIL]": ["[EMAIL]", "[CARD]"],
    }
    assert result.reasons == ("card", "email", "name")


@pytest.mark.parametrize("sample", ['{"x":1,"x":2}', '{"a@b.test":1,"b@b.test":2}'])
def test_collision_blocks(sample: str) -> None:
    """Ambiguous objects must never lose fields during sanitization."""
    result = PiiGuard(Policy()).check(sample, GuardContext())
    assert result.risk_score == 1 and result.sanitized_text is None


def test_length_limit_and_regex_only_names() -> None:
    """Labeled names work offline and oversize output always fails closed."""
    guard = PiiGuard(Policy())
    assert (
        guard.check("My name is Kai Example.", GuardContext()).sanitized_text
        == "My name is [NAME]."
    )
    assert guard.check("x" * 33_000, GuardContext()).reasons == ("output_length_limit",)
    assert guard.check('{"bad": a@example.test', GuardContext()).sanitized_text == '{"bad": [EMAIL]'


def test_lazy_ner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The recognizer loads once and PERSON ranges are merged with regex ranges."""
    loads: list[str] = []

    def fake_load(name: str) -> object:
        """Record loads and identify the first two characters as a person."""
        loads.append(name)
        return lambda text: SimpleNamespace(
            ents=(SimpleNamespace(start_char=0, end_char=2, label_="PERSON"),)
        )

    monkeypatch.setattr(
        "cerberus.guards.pii.importlib.import_module", lambda _: SimpleNamespace(load=fake_load)
    )
    guard = PiiGuard(Policy(output=OutputSettings(ner_enabled=True)))
    assert not loads
    assert guard.check("Jo spoke.", GuardContext()).sanitized_text == "[NAME] spoke."
    assert guard.check("jo@example.test", GuardContext()).sanitized_text == "[EMAIL]"
    assert loads == ["en_core_web_sm"]


@pytest.mark.parametrize("closed", [True, False])
def test_unavailable_ner_preserves_other_findings(
    closed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured failure behavior still applies while regex checks run."""

    def unavailable(_: str) -> object:
        """Simulate a missing optional backend."""
        raise ImportError

    guard = PiiGuard(
        Policy(fail_mode="closed" if closed else "open", output=OutputSettings(ner_enabled=True))
    )
    monkeypatch.setattr(guard, "_persons", unavailable)
    result = guard.check("jo@example.test", GuardContext())
    assert result.reasons == ("email", "ner_unavailable")
    assert result.sanitized_text == (None if closed else "[EMAIL]")
