"""Provider wire contract, credential isolation, and response budget tests."""

import json
from collections.abc import Iterator

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from cerberus.providers import client
from cerberus.providers.client import ProviderConfig, ProviderError, build_provider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deployment settings independent of the operator's local environment."""
    for name in [
        "CERBERUS_PROVIDER",
        "CERBERUS_MODEL",
        "CERBERUS_PROVIDER_URL",
        "CERBERUS_PROVIDER_TIMEOUT",
        "CERBERUS_MAX_TOKENS",
        "CERBERUS_MAX_RESPONSE_BYTES",
        "CERBERUS_MAX_OUTPUT_CHARS",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "kind,path",
    [
        ("openai", "/v1/chat/completions"),
        ("anthropic", "/v1/messages"),
        ("ollama", "/api/chat"),
    ],
)
def test_chat_and_judge_wire_format(kind: str, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve trusted roles, bound output, authenticate only the configured endpoint."""
    monkeypatch.setenv("CERBERUS_PROVIDER", kind)
    monkeypatch.setenv("CERBERUS_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    config = ProviderConfig.from_env()
    requests: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        """Simulate the vendor's text-only response with optional nontext blocks."""
        requests.append(request)
        envelopes = {
            "openai": {"choices": [{"message": {"content": "reply"}}]},
            "anthropic": {"content": [{"type": "thinking"}, {"type": "text", "text": "reply"}]},
            "ollama": {"message": {"content": "reply"}},
        }
        return httpx.Response(200, json=envelopes[kind])

    provider = build_provider(config, transport=httpx.MockTransport(answer))
    try:
        assert provider.complete("system", "user") == "reply"
        assert provider.judge_complete("system", "user") == "reply"
    finally:
        provider.close()
    request = requests[0]
    assert request.url.path == path
    body = json.loads(request.content)
    judge = json.loads(requests[1].content)
    assert body["stream"] is False and body["model"] == "test-model"
    assert body["messages"][-1] == {"role": "user", "content": "user"}
    if kind == "anthropic":
        assert body["system"] == "system" and body["max_tokens"] == config.max_tokens
        assert len(body["messages"]) == 1
        assert request.headers["x-api-key"] == "test-anthropic"
        assert request.headers["anthropic-version"] == "2023-06-01"
    else:
        assert body["messages"][0] == {"role": "system", "content": "system"}
    if kind == "openai":
        assert request.headers["authorization"] == "Bearer test-openai"
        assert body["max_completion_tokens"] == config.max_tokens
        assert judge["response_format"] == {"type": "json_object"}
    if kind == "ollama":
        assert "authorization" not in request.headers and "x-api-key" not in request.headers
        assert body["options"] == {"num_predict": config.max_tokens}
        assert "format" not in body and judge["format"]["type"] == "object"
        assert judge["options"] == {"num_predict": config.max_tokens, "temperature": 0, "seed": 42}


def test_env_and_private_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default local chat is credential-free; local container hosts are supported."""
    assert ProviderConfig.from_env().model == "qwen2.5:0.5b"
    for url in [
        "http://ollama:11434",
        "http://host.docker.internal:11434",
        "http://10.0.1.2:11434",
    ]:
        assert ProviderConfig(base_url=url).base_url == url
    monkeypatch.setenv("CERBERUS_PROVIDER", "unknown")
    with pytest.raises(ValueError):
        ProviderConfig.from_env()
    monkeypatch.setenv("CERBERUS_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        ProviderConfig.from_env()
    monkeypatch.setenv("CERBERUS_PROVIDER_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("CERBERUS_MODEL", "local")
    assert ProviderConfig.from_env().model == "local"


@pytest.mark.parametrize(
    "url",
    [
        "https://admin:password@example.test",
        "https://example.test?q=q",
        "ftp://example.test",
        "https://example.test#fragment",
        "http://example.test",
        "http://8.8.8.8",
        "/relative/path",
        "https://example.test:invalid-port",
    ],
)
def test_bad_urls(url: str) -> None:
    """Provider URLs cannot smuggle credentials or silently use public plaintext HTTP."""
    with pytest.raises(ValidationError):
        ProviderConfig(base_url=url)


def test_credential_validation() -> None:
    """API keys cannot be attached to Ollama or used as header-injection payloads."""
    with pytest.raises(ValidationError):
        ProviderConfig(api_key=SecretStr("credential"))
    with pytest.raises(ValidationError):
        ProviderConfig(kind="anthropic", base_url="https://example.test")
    with pytest.raises(ValidationError) as caught:
        ProviderConfig(
            kind="openai", base_url="https://example.test", api_key=SecretStr("secret\n")
        )
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="sensitive data"),
        httpx.Response(307, headers={"location": "https://sensitive.invalid"}),
        httpx.Response(200, content=b"x", headers={"content-length": "99999999"}),
        httpx.Response(200, text="invalid JSON with sensitive data"),
        httpx.Response(200, json={"message": {"content": 42}}),
        httpx.Response(200, json={"message": {"content": "too long"}}),
        httpx.Response(200, json={"other": "sensitive data"}),
    ],
)
def test_generic_error(response: httpx.Response) -> None:
    """Upstream statuses, invalid bodies and redirects never enter the error text."""
    provider = build_provider(
        ProviderConfig(max_output_chars=3),
        transport=httpx.MockTransport(lambda _: response),
    )
    try:
        with pytest.raises(ProviderError) as caught:
            provider.complete("", "sensitive prompt")
    finally:
        provider.close()
    assert "sensitive" not in str(caught.value)


class Chunks(httpx.SyncByteStream):
    """Synthetic chunks with no declared content length."""

    def __iter__(self) -> Iterator[bytes]:
        """Deliver more than 1 KiB split across two chunks."""
        yield b"x" * 700
        yield b"x" * 700


@pytest.mark.parametrize("encoding", ["identity", "gzip"])
def test_stream_limits(encoding: str) -> None:
    """Unsigned lengths and compressed responses are bounded before decoding."""
    provider = build_provider(
        ProviderConfig(max_response_bytes=1024),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                stream=Chunks(),
                headers={"content-encoding": encoding},
            )
        ),
    )
    try:
        with pytest.raises(ProviderError, match="limit|encoding"):
            provider.complete("", "hello")
    finally:
        provider.close()


def test_slow_response_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """An elapsed whole-response budget stops a drip-fed response."""
    times = iter([1.0, 2.0])
    monkeypatch.setattr(client, "monotonic", lambda: next(times))
    provider = build_provider(
        ProviderConfig(timeout_seconds=0.5),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Chunks())),
    )
    try:
        with pytest.raises(ProviderError, match="timed out"):
            provider.complete("", "hello")
    finally:
        provider.close()
