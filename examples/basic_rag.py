"""Run a small local retrieval and generation flow with python -m examples.basic_rag."""

import re

from cerberus import protect
from cerberus.core import GuardContext

_PASSAGES = (
    "The archive opens on weekdays at 09:00.",
    "For archive access, email the demo desk at archive@example.test.",
)


def retrieve(question: str) -> tuple[str, ...]:
    """Select the passage sharing the most terms with the request."""
    query = set(re.findall(r"\w+", question.casefold())) - {"the", "at", "a", "on", "for"}
    scores = [
        (len(query.intersection(re.findall(r"\w+", doc.casefold()))), doc) for doc in _PASSAGES
    ]
    matches, best = max(scores)
    return (best,) if matches else ()


@protect(policy="strict")
def answer(question: str, *, context: GuardContext) -> str:
    """Use a local extractive response as the example's generation step."""
    if not question.strip() or not context.documents:
        return "The archive has no matching passage."
    return context.documents[0]


def ask(question: str) -> str:
    """Guard both the question and the exact passages provided to generation."""
    return answer(question, context=GuardContext(documents=retrieve(question)))


if __name__ == "__main__":
    print(ask("Where can I email the archive desk for access?"))
