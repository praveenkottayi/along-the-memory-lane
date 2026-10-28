"""
The query pipeline as a LangGraph graph (Roadmap step 0).

Same behaviour as the straight-line pipeline in rag.py — the nodes call the very
same functions — but expressed as a graph, so later steps can add nodes, routes,
loops and memory without rewriting the flow.

    START → expand → search → threshold ─┬─(chunks found)──→ answer → END
                                         └─(nothing found)─→ END

LangGraph concepts used here:
    State             MemoryState — the shared "notebook" every node reads from
                      and writes to. A node returns only the fields it changed;
                      LangGraph merges them into the state.
    Node              a plain function: (state, runtime) -> dict of updates.
    Edge              add_edge(a, b): after a, always run b.
    Conditional edge  add_conditional_edges(a, router): after a, the router
                      function looks at the state and names the next node.
    Runtime context   things nodes need that aren't data — here the LlamaIndex
                      index. Passed per call (`context=`), not stored in the state.
    Compile           turns the graph definition into a runnable object.
    Invoke / stream   run it to the end, or step by step.

Usage:
    from graph import ask
    state = ask(index, "What is my half marathon personal best?")
    state["answer"], state["chunks"], state["fetched"]

    python graph.py            # print the graph as a Mermaid diagram
"""
from dataclasses import dataclass
from typing import Literal, TypedDict

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from rag import (
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    apply_threshold,
    expand_query,
    search,
    synthesize,
)


# ---------------------------------------------------------------------------
# State — everything the pipeline knows about one question
# ---------------------------------------------------------------------------

class MemoryState(TypedDict, total=False):
    # Inputs (set by ask())
    question: str
    top_k: int
    threshold: float
    filters: MetadataFilters | None
    # Written by the nodes
    expanded_query: str              # expand
    fetched: list[NodeWithScore]     # search — all candidates (the eval uses these too)
    chunks: list[NodeWithScore]      # threshold — what the LLM will see
    answer: str                      # answer — "" when nothing passed the threshold


@dataclass
class Deps:
    """Runtime context: dependencies the nodes need, passed in per call."""
    index: VectorStoreIndex


# ---------------------------------------------------------------------------
# Nodes — each reads the state and returns only the fields it updates
# ---------------------------------------------------------------------------

def expand_node(state: MemoryState) -> dict:
    """Rewrite personal shorthand using the glossary (e.g. a nickname → full form)."""
    return {"expanded_query": expand_query(state["question"])}


def search_node(state: MemoryState, runtime: Runtime[Deps]) -> dict:
    """Vector search in ChromaDB: the 2 × top_k closest chunks."""
    fetched = search(runtime.context.index, state["expanded_query"], state["top_k"], state.get("filters"))
    return {"fetched": fetched}


def threshold_node(state: MemoryState) -> dict:
    """Keep chunks scoring >= threshold, best top_k."""
    return {"chunks": apply_threshold(state["fetched"], state["top_k"], state["threshold"])}


def answer_node(state: MemoryState) -> dict:
    """Llama writes the answer from the kept chunks only.

    Uses the original question (not the expanded one), as the app always has.
    """
    return {"answer": synthesize(state["question"], state["chunks"])}


# ---------------------------------------------------------------------------
# Routing — the conditional edge after `threshold`
# ---------------------------------------------------------------------------

def has_chunks(state: MemoryState) -> Literal["answer", "__end__"]:
    """Nothing passed the threshold → stop without calling the LLM."""
    return "answer" if state["chunks"] else END


# ---------------------------------------------------------------------------
# Build + compile
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(MemoryState, context_schema=Deps)

    graph.add_node("expand", expand_node)
    graph.add_node("search", search_node)
    graph.add_node("threshold", threshold_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "expand")
    graph.add_edge("expand", "search")
    graph.add_edge("search", "threshold")
    graph.add_conditional_edges("threshold", has_chunks, ["answer", END])
    graph.add_edge("answer", END)

    return graph.compile()


# Compiled once at import; the index is supplied per call, so one graph serves
# the app, the eval and any script.
GRAPH = build_graph()


def initial_state(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    filters: MetadataFilters | None = None,
) -> MemoryState:
    return {"question": question, "top_k": top_k, "threshold": threshold, "filters": filters}


def ask(
    index: VectorStoreIndex,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    filters: MetadataFilters | None = None,
) -> MemoryState:
    """Run the whole graph for one question and return the final state."""
    state = GRAPH.invoke(initial_state(question, top_k, threshold, filters), context=Deps(index=index))
    state.setdefault("answer", "")  # the "nothing found" path never runs the answer node
    return state


def ask_stream(
    index: VectorStoreIndex,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    filters: MetadataFilters | None = None,
):
    """Run the graph step by step: yields (node name, that node's updates)."""
    for update in GRAPH.stream(
        initial_state(question, top_k, threshold, filters),
        context=Deps(index=index),
        stream_mode="updates",
    ):
        for node, changes in update.items():
            yield node, changes


def mermaid() -> str:
    """The graph as a Mermaid diagram, generated from the code itself."""
    return GRAPH.get_graph().draw_mermaid()


if __name__ == "__main__":
    print(mermaid())
