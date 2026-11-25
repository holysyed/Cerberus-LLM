"""Decorators for application-owned, non-streaming text functions."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from functools import wraps
from pathlib import Path
from typing import Literal, TypeVar, cast

from cerberus.core.audit import AuditStore
from cerberus.core.models import GuardContext, PipelineResult
from cerberus.core.policy import Policy, load_policy
from cerberus.guards.factory import Guardrails

F = TypeVar("F", bound=Callable[..., object])
Stage = Literal["input", "output"]


class GuardViolation(Exception):
    """A policy block carrying only its stage and request correlation ID."""

    def __init__(self, stage: Stage, request_id: str) -> None:
        """Build a client-safe exception without raw text or provider diagnostics."""
        self.stage = stage
        self.request_id = request_id
        super().__init__(f"Cerberus blocked {stage} for request {request_id}")


def _release(result: PipelineResult, stage: Stage) -> str:
    """Release the final pipeline text or raise a content-free policy exception."""
    if result.blocked:
        raise GuardViolation(stage, result.request_id)
    return result.text


def protect(
    *,
    policy: str | Path | Policy = "strict",
    audit_store: AuditStore | None = None,
    hash_key: bytes | None = None,
    classifier: Callable[[str], float] | None = None,
    judge: Callable[[str], float] | None = None,
) -> Callable[[F], F]:
    """Guard a sync or async function taking one string and optional named context.

    The text parameter can be positional-only, positional-or-keyword, or keyword-only.
    An additional parameter named context accepts a trusted GuardContext. A missing
    or explicitly None context gets a fresh request ID. The function must return
    one complete string and is invoked at most once per call.
    """
    configured = policy if isinstance(policy, Policy) else load_policy(policy)
    rails = Guardrails.build(
        configured, audit_store=audit_store, hash_key=hash_key, classifier=classifier, judge=judge
    )

    def decorate(func: F) -> F:
        """Preserve the original callable signature for framework integration."""
        signature = inspect.signature(func)
        parameters = signature.parameters
        names = [name for name in parameters if name != "context"]
        if (
            len(names) != 1
            or any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters.values())
            or inspect.isgeneratorfunction(func)
            or inspect.isasyncgenfunction(func)
        ):
            raise TypeError(
                "protect requires one text parameter, optional context, and a final string"
            )
        text_name = names[0]

        def prepare(
            args: tuple[object, ...], kwargs: dict[str, object]
        ) -> tuple[inspect.BoundArguments, GuardContext]:
            """Bind all supplied data and replace text with the guarded input."""
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            value = bound.arguments.get("context")
            context = GuardContext() if value is None else value
            if not isinstance(context, GuardContext):
                raise TypeError("context must be a GuardContext")
            text = bound.arguments[text_name]
            if not isinstance(text, str):
                raise TypeError("protected input must be a string")
            bound.arguments[text_name] = _release(rails.inputs.run(text, context), "input")
            if "context" in parameters:
                bound.arguments["context"] = context
            return bound, context

        def finish(value: object, context: GuardContext) -> str:
            """Validate the final output once, regardless of schema repair settings."""
            if not isinstance(value, str):
                raise TypeError("protected output must be a string")
            return _release(rails.outputs.run(value, context), "output")

        if inspect.iscoroutinefunction(func):
            async_func = cast(Callable[..., Awaitable[object]], func)

            @wraps(func)
            async def async_wrapper(*args: object, **kwargs: object) -> str:
                """Keep synchronous guard inference off the caller's event loop."""
                bound, context = await asyncio.to_thread(prepare, args, kwargs)
                output = await async_func(*bound.args, **bound.kwargs)
                return await asyncio.to_thread(finish, output, context)

            return cast(F, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: object, **kwargs: object) -> str:
            """Run both guard stages in the caller's execution context."""
            bound, context = prepare(args, kwargs)
            return finish(func(*bound.args, **bound.kwargs), context)

        return cast(F, sync_wrapper)

    return decorate
