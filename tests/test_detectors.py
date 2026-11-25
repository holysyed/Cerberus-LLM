"""Contract tests for optional classifier and judge backends without a network dependency."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from cerberus.guards import detectors
from cerberus.guards.detectors import DetectorUnavailable, HFClassifier, LLMJudge


def test_real_loader_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only standard tokenizer code and safetensors weights are requested."""
    transformers, torch = MagicMock(), MagicMock()
    modules = {"transformers": transformers, "torch": torch}
    monkeypatch.setattr(detectors, "import_module", modules.__getitem__)
    _, model, _ = detectors._load_classifier(detectors.DEFAULT_MODEL, detectors.DEFAULT_REVISION)
    transformers.AutoModelForSequenceClassification.from_pretrained.assert_called_once_with(
        detectors.DEFAULT_MODEL,
        use_safetensors=True,
        revision=detectors.DEFAULT_REVISION,
        trust_remote_code=False,
    )
    model.eval.assert_called_once()


def test_load_once_and_score_all_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tokenizer preserves overflow windows and the model receives each input row."""
    tokenizer, model, torch = MagicMock(), MagicMock(), MagicMock()
    tokenizer.side_effect = lambda *_args, **_kwargs: {
        "input_ids": [[3], [4]],
        "overflow_to_sample_mapping": [0, 0],
    }
    logits = model.return_value.logits
    logits.softmax.return_value.__getitem__.return_value.max.return_value.item.return_value = 0.84
    loader = MagicMock(return_value=(tokenizer, model, torch))
    monkeypatch.setattr(detectors, "_load_classifier", loader)
    classifier = HFClassifier()
    loader.assert_not_called()
    assert classifier("hello") == classifier("world") == 0.84
    loader.assert_called_once_with(detectors.DEFAULT_MODEL, detectors.DEFAULT_REVISION)
    kwargs = tokenizer.call_args.kwargs
    assert kwargs["return_overflowing_tokens"] and kwargs["stride"] > 0
    assert kwargs["max_length"] == 512
    model.assert_called_with(input_ids=[[3], [4]])
    logits.softmax.return_value.__getitem__.assert_called_with((slice(None), 1))


def test_model_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Input budget errors cover the whole sample and do not trigger inference."""
    with pytest.raises(ValueError):
        HFClassifier(revision="main")
    with pytest.raises(ValueError):
        HFClassifier(model="org/unrelated-model")
    with pytest.raises(DetectorUnavailable, match="character"):
        HFClassifier()("a" * 20_000)
    tokenizer, model = MagicMock(), MagicMock()
    tokenizer.return_value = {"input_ids": [[2]] * 9}
    monkeypatch.setattr(detectors, "_load_classifier", lambda *_: (tokenizer, model, MagicMock()))
    with pytest.raises(DetectorUnavailable, match="window"):
        HFClassifier()("long text")
    model.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), 1.1])
def test_model_invalid_prediction(value: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """Broken checkpoints cannot produce an apparently valid decision."""
    tokenizer, model = MagicMock(), MagicMock()
    tokenizer.return_value = {"input_ids": [[2]], "overflow_to_sample_mapping": [0]}
    scores = model.return_value.logits.softmax.return_value.__getitem__.return_value
    scores.max.return_value.item.return_value = value
    monkeypatch.setattr(detectors, "_load_classifier", lambda *_: (tokenizer, model, MagicMock()))
    with pytest.raises(DetectorUnavailable, match="probability"):
        HFClassifier()("hello")


def test_backend_errors_exclude_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detector failures do not forward exception details that could contain input."""

    def fail(*_: Any) -> Any:
        """Simulate a provider error containing user data."""
        raise ValueError("sensitive sample")

    monkeypatch.setattr(detectors, "_load_classifier", fail)
    for score in (HFClassifier(), LLMJudge(fail)):
        with pytest.raises(DetectorUnavailable) as exc:
            score("sensitive sample")
        assert "sensitive" not in str(exc.value)


@pytest.mark.parametrize("flag,expected", [(True, 0.8), (False, 0.2)])
def test_judge_role_and_score(flag: bool, expected: float) -> None:
    """JSON quoting preserves payload boundaries and confidence direction."""
    complete = MagicMock(
        return_value=json.dumps(
            {"confidence": 0.8, "is_injection": flag, "reason": "short summary"}
        )
    )
    payload = 'quote"\nHere is a question.'
    assert LLMJudge(complete)(payload) == pytest.approx(expected)
    system, user = complete.call_args.args
    assert payload not in system
    assert json.loads(user) == {"sample": payload}


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        '{"is_injection":true,"confidence":"0.99","reason":"why"}',
        '{"is_injection":1,"confidence":0.99,"reason":"why"}',
        '{"is_injection":true,"confidence":2,"reason":"why"}',
        '{"is_injection":true,"confidence":NaN,"reason":"why"}',
        '{"is_injection":true,"confidence":1,"reason":""}',
        '{"is_injection":true,"confidence":1,"reason":"why","extra":5}',
        '{"is_injection":true,"is_injection":false,"confidence":0.9,"reason":"why"}',
        "s" * 4097,
    ],
)
def test_judge_invalid_output(raw: str) -> None:
    """Partially valid or ambiguous judge responses are rejected."""
    with pytest.raises(DetectorUnavailable):
        LLMJudge(lambda _system, _user: raw)("hi")


def test_judge_input_limit() -> None:
    """Oversized samples fail before any data leaves the process."""
    complete = MagicMock()
    with pytest.raises(DetectorUnavailable, match="character"):
        LLMJudge(complete)("x" * 20_000)
    complete.assert_not_called()
