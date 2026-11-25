"""Use the protected RAG function inside a LangChain runnable and a LangGraph node."""

from typing import TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from examples.basic_rag import ask


class State(TypedDict):
    """State shared between the graph's caller and its answer node."""

    question: str
    answer: str


def run(question: str) -> str:
    """Execute the local chain inside a compiled graph and return guarded output."""
    chain = RunnableLambda(ask)

    def generate(state: State) -> dict[str, str]:
        """Use the runnable's guarded answer as the graph update."""
        return {"answer": chain.invoke(state["question"])}

    graph = (
        StateGraph(State)
        .add_node("generate", generate)
        .add_edge(START, "generate")
        .add_edge("generate", END)
        .compile()
    )
    return str(graph.invoke({"question": question, "answer": ""})["answer"])


if __name__ == "__main__":
    print(run("When does the archive open on weekdays?"))
