"""
agents/insight.py
Answers natural language questions about the user's logged activity and
generates weekly digests. Routes each question to structured SQL-style
lookups, semantic search, or both, then synthesises an answer with Claude.
"""

from datetime import date, timedelta
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from config import settings
from agents.memory import get_entries, semantic_search, get_week_stats

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ── Query routing ──────────────────────────────────────────────────────────

class QueryRoute(BaseModel):
    strategy: Literal["structured", "semantic", "both"] = Field(
        description=(
            "'structured' for counts/dates/ratings ('how many movies last month'), "
            "'semantic' for similarity/theme questions ('find something like X'), "
            "'both' when the question needs filtering AND meaning."
        )
    )
    entry_type: Literal["movie", "book", "game", "social", "any"] = Field(
        description="Which entry type the question is about, or 'any'."
    )
    search_terms: str = Field(description="Best search phrase to use for semantic lookup.")


ROUTE_TOOL = {
    "name": "route_query",
    "description": "Decide how to answer a question about the user's logged activity.",
    "input_schema": QueryRoute.model_json_schema(),
}


def _route_query(question: str) -> QueryRoute:
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=256,
        system="Decide the best retrieval strategy for this question about a personal activity log.",
        tools=[ROUTE_TOOL],
        tool_choice={"type": "tool", "name": "route_query"},
        messages=[{"role": "user", "content": question}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return QueryRoute.model_validate(tool_block.input)


# ── Answering ──────────────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    """
    Main entry point for the 'Ask anything' Streamlit page.
    Routes the question, retrieves context, and asks Claude to synthesise
    a grounded answer — never lets the model answer from memory alone.
    """
    route = _route_query(question)

    context_parts = []

    if route.strategy in ("structured", "both"):
        entry_type = None if route.entry_type == "any" else route.entry_type
        entries = get_entries(entry_type=entry_type, limit=30)
        context_parts.append(
            "Structured records:\n" +
            "\n".join(f"- {e.entry_type.value}: {e.title} ({e.event_date}, rating={e.rating})" for e in entries)
        )

    if route.strategy in ("semantic", "both"):
        entry_type = None if route.entry_type == "any" else route.entry_type
        hits = semantic_search(route.search_terms, n_results=8, entry_type=entry_type)
        context_parts.append(
            "Semantically similar entries:\n" +
            "\n".join(f"- {h['document']}" for h in hits)
        )

    context = "\n\n".join(context_parts) if context_parts else "No matching records found."

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        system=(
            "You answer questions about the user's personal activity log (movies, books, "
            "games, social events). Use ONLY the provided context. If the context doesn't "
            "answer the question, say so plainly rather than guessing."
        ),
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        }],
    )

    return "".join(b.text for b in response.content if b.type == "text")


# ── Weekly digest ────────────────────────────────────────────────────────

def generate_weekly_digest(week_start: date) -> dict:
    """
    Called by the scheduler every Monday for the prior week.
    Returns {"summary": str, "stats": dict} ready to store as a Digest row.
    """
    stats = get_week_stats(week_start)

    if stats["total_entries"] == 0:
        return {
            "summary": "No activity logged this week.",
            "stats": {k: v for k, v in stats.items() if k != "entries"},
        }

    entry_lines = "\n".join(
        f"- {e.entry_type.value}: {e.title} (mood={e.mood_score}, tags={e.tags})"
        for e in stats["entries"]
    )

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=512,
        system=(
            "Write a warm, brief (3-5 sentence) weekly reflection on the user's logged "
            "activity. Mention patterns in mood or theme if they stand out. Second person, "
            "conversational tone — like a friend noticing what you've been into lately."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Week of {week_start.isoformat()}\n"
                f"Total entries: {stats['total_entries']}\n"
                f"By type: {stats['counts_by_type']}\n"
                f"Average mood: {stats['avg_mood']}\n"
                f"Top tags: {stats['top_tags']}\n\n"
                f"Entries:\n{entry_lines}"
            ),
        }],
    )

    summary = "".join(b.text for b in response.content if b.type == "text")

    return {
        "summary": summary,
        "stats": {k: v for k, v in stats.items() if k != "entries"},
    }
