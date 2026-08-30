"""
agents/enrichment.py
Fetches external metadata (TMDB, IGDB) and, for entries that came from
raw imports without LLM-parsed tags (Letterboxd, Goodreads), generates
tags and a mood score via Claude.

Runs *after* ingestion — decoupled so a failed enrichment call never
blocks storing the underlying entry.
"""

import logging
from typing import Annotated, Optional

import anthropic
from annotated_types import Ge, Le
from pydantic import BaseModel, Field

from config import settings
from db.models import Entry, EntryType, SessionLocal
from tools.tmdb import fetch_movie_metadata
from tools.igdb import fetch_game_metadata
from agents.memory import attach_metadata, update_entry_enrichment

logger = logging.getLogger(__name__)
client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ── Tagging schema (same tool-calling pattern as tools/parsers.py) ─────────

class TagResult(BaseModel):
    tags: list[str] = Field(min_length=1, max_length=6)
    mood_score: Annotated[float, Ge(-1), Le(1)]


TAG_TOOL = {
    "name": "tag_entry",
    "description": "Generate thematic tags and a mood score for a logged entry.",
    "input_schema": TagResult.model_json_schema(),
}


def _generate_tags(title: str, context: str) -> TagResult:
    """Used for entries imported without LLM parsing (Letterboxd/Goodreads)."""
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=256,
        system=(
            "Generate 3-6 lowercase thematic tags and a mood score (-1 to 1) "
            "for this entry. Prefer specific tags over generic ones."
        ),
        tools=[TAG_TOOL],
        tool_choice={"type": "tool", "name": "tag_entry"},
        messages=[{"role": "user", "content": f"{title}\n\n{context}"}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return TagResult.model_validate(tool_block.input)


# ── Per-entry enrichment ─────────────────────────────────────────────────

def enrich_entry(entry_id: str) -> dict:
    """
    Enrich a single entry: fetch external metadata (movie/game only),
    and generate tags/mood if they weren't already set by the parser.
    """
    db = SessionLocal()
    try:
        entry = db.query(Entry).filter_by(id=entry_id).first()
        if not entry:
            return {"status": "not_found", "entry_id": entry_id}

        result = {"entry_id": entry_id, "metadata": False, "tags": False}

        # Step 1: external metadata for movies and games
        meta = None
        if entry.entry_type == EntryType.movie:
            meta = fetch_movie_metadata(entry.title, entry.event_date.year if entry.event_date else None)
        elif entry.entry_type == EntryType.game:
            meta = fetch_game_metadata(entry.title)

        if meta:
            attach_metadata(entry_id, meta, db=db)
            result["metadata"] = True

        # Step 2: tags/mood — only if the entry doesn't already have them
        # (manual journal entries already got these from tools/parsers.py)
        if not entry.tags or entry.mood_score is None:
            context = entry.raw_text or (meta.get("extra", {}).get("summary") if meta else "") or ""
            try:
                tag_result = _generate_tags(entry.title, context)
                update_entry_enrichment(entry_id, tag_result.tags, tag_result.mood_score, db=db)
                result["tags"] = True
            except Exception:  # noqa: BLE001
                logger.exception("Tag generation failed for entry %s", entry_id)

        return {"status": "ok", **result}

    finally:
        db.close()


def enrich_batch(entry_ids: list[str]) -> list[dict]:
    """Enrich multiple entries — used after a bulk sync (Letterboxd/Goodreads)."""
    return [enrich_entry(eid) for eid in entry_ids]
