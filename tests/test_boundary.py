"""Exercise the gateway's authentication, input limits, and correlation boundary."""

from collections.abc import Iterator
from pathlib import Path
from secrets import token_urlsafe
from uuid import UUID

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from cerberus.gateway.security import BoundaryMiddleware, Settings


def boundary(limit: int = 32, rpm: int = 60) -> tuple[TestClient, dict[str, str]]:
    """Create an isolated protected route and fresh credential."""
    app = FastAPI()
    key = token_urlsafe(32)
    settings = Settings(api_key=SecretStr(key), max_body_bytes=limit, requests_per_minute=rpm)
    app.add_middleware(BoundaryMiddleware, settings=settings)

    @app.post("/v1/sample")
    async def sample(request: Request) -> dict[str, int]:
        """Report the size consumed by the downstream application."""
        return {"bytes": len(await request.body())}

    @app.get("/v1/error")
    async def broken() -> None:
        """Simulate a backend exception carrying private text."""
        raise RuntimeError("PRIVATE_TEST_CONTENT")

    return TestClient(app), {"X-API-Key": key}


def test_boundary_authenticates_before_parsing_and_rate_limiting() -> None:
    """An unauthorized request consumes zero authenticated quota."""
    client, headers = boundary(rpm=1)
    first = client.post("/v1/sample", content=b"x" * 50)
    assert first.status_code == 401
    assert UUID(first.headers["X-Request-ID"])
    good = client.post("/v1/sample", headers=headers, content=b"ab")
    assert good.json() == {"bytes": 2}
    limited = client.post("/v1/sample", headers=headers)
    assert limited.status_code == 429 and limited.headers["Retry-After"] == "60"
    assert limited.headers["X-Request-ID"] != good.headers["X-Request-ID"]


def test_declared_and_chunked_limits() -> None:
    """Bound real bytes even when the client omits Content-Length."""
    client, headers = boundary(limit=4)
    assert client.post("/v1/sample", headers=headers, content=b"12345").status_code == 413

    def chunks() -> Iterator[bytes]:
        """Generate a chunked request exceeding the same bound."""
        yield b"123"
        yield b"45"

    assert client.post("/v1/sample", headers=headers, content=chunks()).status_code == 413
    assert client.post("/v1/sample", headers={**headers, "Content-Length": "x"}).status_code == 400
    response = client.get("/v1/error", headers=headers)
    assert response.status_code == 500 and "PRIVATE_TEST_CONTENT" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Application schema and credentials come from deployment configuration."""
    monkeypatch.delenv("CERBERUS_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings.from_env()
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"string"}', encoding="utf-8")
    key = token_urlsafe(32)
    monkeypatch.setenv("CERBERUS_API_KEY", key)
    monkeypatch.setenv("CERBERUS_SCHEMA_FILE", str(schema))
    settings = Settings.from_env()
    assert settings.output_schema == {"type": "string"} and key not in repr(settings)
