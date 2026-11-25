"""Bounded synchronous chat adapters for administrator-selected LLM endpoints."""

from __future__ import annotations

import ipaddress
import json
import os
from time import monotonic
from typing import Any, Literal, Protocol, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from cerberus.guards.detectors import JudgeVerdict

_DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama": "http://127.0.0.1:11434",
}
_PATHS = {"openai": "chat/completions", "anthropic": "messages", "ollama": "api/chat"}


class ProviderError(RuntimeError):
    """A provider failed to return a valid, bounded text completion."""


class Provider(Protocol):
    """Minimum application provider interface for ordinary guarded chat."""

    def complete(self, system: str, user: str) -> str:
        """Generate a response preserving separate trusted and user message roles."""
        ...


class ProviderConfig(BaseModel):
    """Server-owned endpoint, credentials, and request budgets."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["openai", "anthropic", "ollama"] = "ollama"
    base_url: str = _DEFAULT_URLS["ollama"]
    model: str = Field(default="qwen2.5:0.5b", min_length=1, max_length=256)
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=60, ge=0.1, le=600, allow_inf_nan=False)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    max_response_bytes: int = Field(default=524_288, ge=1024, le=4_194_304)
    max_output_chars: int = Field(default=65_536, ge=1, le=131_072)

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        """Reject URL credentials, ambiguous endpoints, and cleartext remote secrets."""
        try:
            url = httpx.URL(self.base_url)
        except httpx.InvalidURL:
            raise ValueError("Invalid provider base URL") from None
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
        ):
            raise ValueError("Expected an HTTP(S) provider base URL without credentials or query")
        key = self.api_key.get_secret_value() if self.api_key else ""
        if self.kind == "ollama" and key:
            raise ValueError("Ollama uses an unauthenticated deployment endpoint")
        if (self.kind == "anthropic" or url.host == "api.openai.com") and not key:
            raise ValueError("This provider requires an API key")
        if key and (not key.isascii() or not key.isprintable() or key.strip() != key):
            raise ValueError("Invalid provider API key")
        if url.scheme == "http":
            try:
                host_ip = ipaddress.ip_address(url.host)
            except ValueError:
                loopback = url.host == "localhost"
                private = "." not in url.host or url.host == "host.docker.internal"
            else:
                loopback = host_ip.is_loopback
                private = host_ip.is_private
            if not (loopback or (self.kind == "ollama" and private)):
                raise ValueError("Remote provider endpoints require HTTPS")
        return self

    @classmethod
    def from_env(cls) -> Self:
        """Read deployment settings without contacting the selected provider."""
        kind = os.environ.get("CERBERUS_PROVIDER", "ollama")
        if kind not in _DEFAULT_URLS:
            raise ValueError("Unsupported CERBERUS_PROVIDER")
        key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(kind)
        return cls.model_validate(
            {
                "kind": kind,
                "model": os.environ.get(
                    "CERBERUS_MODEL", "qwen2.5:0.5b" if kind == "ollama" else ""
                ),
                "base_url": os.environ.get("CERBERUS_PROVIDER_URL", _DEFAULT_URLS[kind]),
                "api_key": os.environ.get(key_var) if key_var else None,
                "timeout_seconds": os.environ.get("CERBERUS_PROVIDER_TIMEOUT", "60"),
                "max_tokens": os.environ.get("CERBERUS_MAX_TOKENS", "512"),
                "max_response_bytes": os.environ.get("CERBERUS_MAX_RESPONSE_BYTES", "524288"),
                "max_output_chars": os.environ.get("CERBERUS_MAX_OUTPUT_CHARS", "65536"),
            }
        )


class HTTPProvider:
    """Reusable synchronous provider with explicit role separation and finite response budgets."""

    def __init__(
        self, config: ProviderConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        """Build a connection pool with fixed destination and credentials."""
        self.config = config
        headers = {"Accept-Encoding": "identity"}
        if config.kind == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        if config.api_key:
            key = config.api_key.get_secret_value()
            if config.kind == "anthropic":
                headers["x-api-key"] = key
            else:
                headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/") + "/",
            headers=headers,
            transport=transport,
            timeout=config.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        """Release idle network connections."""
        self._client.close()

    def complete(self, system: str, user: str) -> str:
        """Return an ordinary text completion."""
        return self._generate(system, user, judge=False)

    def judge_complete(self, system: str, user: str) -> str:
        """Request a judge verdict, enabling structured generation where supported."""
        return self._generate(system, user, judge=True)

    def _body(self, system: str, user: str, judge: bool) -> dict[str, Any]:
        """Construct the selected provider's native request."""
        messages = [{"role": "user", "content": user}]
        body: dict[str, Any] = {"model": self.config.model, "messages": messages, "stream": False}
        if self.config.kind == "anthropic":
            body.update(system=system, max_tokens=self.config.max_tokens)
        else:
            messages.insert(0, {"role": "system", "content": system})
        if self.config.kind == "openai":
            body["max_completion_tokens"] = self.config.max_tokens
            if judge:
                body["response_format"] = {"type": "json_object"}
        if self.config.kind == "ollama":
            options = {"num_predict": self.config.max_tokens}
            if judge:
                options.update(temperature=0, seed=42)
                body["format"] = JudgeVerdict.model_json_schema()
            body["options"] = options
        return body

    def _generate(self, system: str, user: str, *, judge: bool) -> str:
        """Stream a bounded wire response and extract only the assistant's text."""
        deadline = monotonic() + self.config.timeout_seconds
        try:
            with self._client.stream(
                "POST", _PATHS[self.config.kind], json=self._body(system, user, judge)
            ) as response:
                response.raise_for_status()
                # An upstream cannot force unbounded decompression before length checks.
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise ProviderError("Unsupported provider response encoding")
                if (
                    int(response.headers.get("content-length", "0"))
                    > self.config.max_response_bytes
                ):
                    raise ProviderError("Provider response exceeded its size limit")
                payload = bytearray()
                for chunk in response.iter_bytes():
                    if len(payload) + len(chunk) > self.config.max_response_bytes:
                        raise ProviderError("Provider response exceeded its size limit")
                    if monotonic() > deadline:
                        raise ProviderError("Provider request timed out")
                    payload.extend(chunk)
            result = self._extract(json.loads(payload))
            if (
                not isinstance(result, str)
                or not result
                or len(result) > self.config.max_output_chars
            ):
                raise ProviderError("Provider returned invalid or oversized text")
            return result
        except (
            httpx.HTTPError,
            httpx.InvalidURL,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
            RecursionError,
            RuntimeError,
        ) as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError("Provider request failed or returned an invalid response") from None

    def _extract(self, data: Any) -> object:
        """Decode text from the selected provider's response envelope."""
        if self.config.kind == "openai":
            return data["choices"][0]["message"]["content"]
        if self.config.kind == "anthropic":
            return "\n".join(block["text"] for block in data["content"] if block["type"] == "text")
        return data["message"]["content"]


def build_provider(
    config: ProviderConfig, *, transport: httpx.BaseTransport | None = None
) -> HTTPProvider:
    """Construct a synchronous provider from deployment settings."""
    return HTTPProvider(config, transport=transport)
