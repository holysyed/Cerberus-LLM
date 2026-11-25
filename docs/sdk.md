# Protect an application function

`@cerberus.protect(policy="strict")` accepts a synchronous function or an
`async def` function with one text parameter and a string result. Calls can use
positional arguments, keyword arguments, keyword-only parameters, and defaults.
An optional second parameter named `context` carries the application's trusted
`GuardContext`. Pass retrieved passages and tool responses there before
generation. Each decorated invocation executes the application at most once.

```python
from cerberus import GuardViolation, protect
from cerberus.core import GuardContext

@protect(policy="strict")
def summarize(question: str, *, context: GuardContext) -> str:
    return context.documents[0]

try:
    answer = summarize("When are you open?", context=GuardContext(documents=("Weekdays.",)))
except GuardViolation as error:
    print(error.stage, error.request_id)
```

Use a small adapter around functions with more application parameters, streaming
generators, or structured provider return objects. Extract the final text in that
adapter. For an application-owned JSON response schema, pass
`GuardContext(output_schema=...)`. Set the system prompt in the same context to
enable literal leakage checks. Keep this metadata under application control.

The decorator accepts a packaged policy name, a trusted YAML path, or a validated
`Policy` instance. Supply `audit_store`, `hash_key`, `classifier`, or `judge` when
your policy uses those integrations. A blocked call raises `GuardViolation` with
the stage and correlation ID. Async functions run guard inference in worker
threads and await the application on the caller's event loop.

## Local examples

```sh
python -m examples.basic_rag
python -m pip install -e ".[examples]"
python -m examples.langchain_graph
```

The first example searches two local archive passages and returns the matching
passage through both guard stages. The second wraps that operation in a
LangChain runnable and invokes it from a LangGraph node. Both examples run
without provider credentials. [RunnableLambda reference](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableLambda) · [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
