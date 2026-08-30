"""
agents/orchestrator.py
The LangGraph agent graph. This is the entry point that ties ingestion,
enrichment, memory, and insight together into a single stateful workflow.

Graph shape:

    START
      │
      ▼
   classify_input ──► log_entry ──► needs_enrichment? ──► enrich ──► respond
      │                                    │
      │                                    └──► (no) ────────────► respond
      │
      ├──► answer_query ──────────────────────────────────────────► respond
      │
      └──► sync_sources ───────────────────────────────────────────► respond
                                                                        │
                                                                        ▼
                                                                       END

Why LangGraph instead of a plain chain: the workflow branches based on
what the user is trying to do (log something vs. ask a question vs.
trigger a sync), and the "log_entry" path itself branches again on
whether enrichment is needed. A linear chain can't express this — a
graph with conditional edges can, and it keeps each node independently
testable.
"""

import logging
from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.ingestion import ingest_journal_text, sync_letterboxd, sync_goodreads
from agents.enrichment import enrich_entry
from agents.insight import answer_question

logger = logging.getLogger(__name__)


# ── Graph state ─────────────────────────────────────────────────────────────
# TypedDict is LangGraph's convention for state — every node reads from and
# writes back into this shared dict as it flows through the graph.

class GraphState(TypedDict, total=False):
    user_input: str
    intent: Literal["log", "query", "sync"]
    sync_target: Literal["letterboxd", "goodreads", "all"]

    # populated as the graph runs
    entry_id: Optional[str]
    needs_enrichment: bool
    response: str


# ── Nodes ────────────────────────────────────────────────────────────────────

def classify_input(state: GraphState) -> GraphState:
    """
    Route the user's message to the right path. Kept as simple keyword
    routing rather than an LLM call — the Streamlit UI already tells us
    the intent via which page/button the user used (see run_* helpers
    below), so this node mainly exists for the CLI/chat entry point.
    """
    text = state["user_input"].lower()
    if text.startswith("sync:"):
        target = text.split(":", 1)[1].strip() or "all"
        return {**state, "intent": "sync", "sync_target": target}  # type: ignore[typeddict-item]
    if text.startswith("ask:") or text.strip().endswith("?"):
        return {**state, "intent": "query"}
    return {**state, "intent": "log"}


def log_entry_node(state: GraphState) -> GraphState:
    """Parse + store a freeform journal entry."""
    text = state["user_input"]
    if text.lower().startswith("log:"):
        text = text.split(":", 1)[1].strip()

    result = ingest_journal_text(text)

    if result["status"] == "duplicate":
        return {**state, "response": result["message"], "needs_enrichment": False}

    return {
        **state,
        "entry_id": result["entry_id"],
        "needs_enrichment": result["needs_enrichment"],
        "response": f"Logged: {result['title']} ({result['entry_type']})",
    }


def enrich_node(state: GraphState) -> GraphState:
    """Fetch external metadata + tags for the entry just logged."""
    entry_id = state.get("entry_id")
    if not entry_id:
        return state

    result = enrich_entry(entry_id)
    suffix = " · enriched" if result.get("metadata") or result.get("tags") else ""
    return {**state, "response": state["response"] + suffix}


def query_node(state: GraphState) -> GraphState:
    """Answer a natural language question via the insight agent."""
    text = state["user_input"]
    if text.lower().startswith("ask:"):
        text = text.split(":", 1)[1].strip()

    answer = answer_question(text)
    return {**state, "response": answer}


def sync_node(state: GraphState) -> GraphState:
    """Trigger a source sync (Letterboxd / Goodreads / both)."""
    target = state.get("sync_target", "all")
    results = []

    if target in ("letterboxd", "all"):
        results.append(sync_letterboxd())
    if target in ("goodreads", "all"):
        results.append(sync_goodreads())

    summary = " · ".join(
        f"{r.get('source', 'unknown')}: {r.get('stored', 0)} new" for r in results
    )
    return {**state, "response": f"Sync complete — {summary}"}


# ── Conditional routing ──────────────────────────────────────────────────────

def route_by_intent(state: GraphState) -> str:
    return {"log": "log_entry", "query": "query", "sync": "sync"}[state["intent"]]


def route_after_log(state: GraphState) -> str:
    return "enrich" if state.get("needs_enrichment") else END


# ── Graph assembly ───────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_input", classify_input)
    graph.add_node("log_entry", log_entry_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("query", query_node)
    graph.add_node("sync", sync_node)

    graph.set_entry_point("classify_input")

    graph.add_conditional_edges(
        "classify_input",
        route_by_intent,
        {"log_entry": "log_entry", "query": "query", "sync": "sync"},
    )

    graph.add_conditional_edges(
        "log_entry",
        route_after_log,
        {"enrich": "enrich", END: END},
    )

    graph.add_edge("enrich", END)
    graph.add_edge("query", END)
    graph.add_edge("sync", END)

    return graph.compile()


# Compiled graph — import this everywhere instead of rebuilding it
orchestrator = build_graph()


# ── Convenience entry points (what the Streamlit app actually calls) ────────
# These skip classify_input's keyword-guessing and go straight to the
# intended node, since the UI already knows the user's intent from context.

def run_log(text: str) -> str:
    state: GraphState = {"user_input": f"log: {text}", "intent": "log"}
    result = orchestrator.invoke(state)
    return result["response"]


def run_query(question: str) -> str:
    state: GraphState = {"user_input": f"ask: {question}", "intent": "query"}
    result = orchestrator.invoke(state)
    return result["response"]


def run_sync(target: Literal["letterboxd", "goodreads", "all"] = "all") -> str:
    state: GraphState = {"user_input": f"sync: {target}", "intent": "sync", "sync_target": target}
    result = orchestrator.invoke(state)
    return result["response"]


if __name__ == "__main__":
    # Quick smoke test: python -m agents.orchestrator
    # Prints the graph structure as Mermaid — paste into
    # https://mermaid.live to visualize it.
    print(orchestrator.get_graph().draw_mermaid())
