"""
agents/ingestion.py
Pulls data from external sources (Letterboxd, Goodreads, journal text)
and hands normalised entries to the memory agent for storage.

This agent is deliberately "dumb" — it doesn't decide what to enrich or
how to tag things. It only fetches and stores. Enrichment is a separate
concern handled by agents/enrichment.py, kept separate so either can be
retried or run independently.
"""

import logging
from typing import Optional

from tools.letterboxd import fetch_letterboxd_entries
from tools.igdb import import_goodreads_csv
from tools.parsers import parse_journal_entry
from agents.memory import store_entry

logger = logging.getLogger(__name__)


def ingest_journal_text(text: str) -> dict:
    """
    Entry point for the Streamlit journal box. Parses freeform text with
    Claude, stores it, and returns a result dict the UI can render.
    """
    parsed = parse_journal_entry(text)
    entry = store_entry(parsed)

    if entry is None:
        return {"status": "duplicate", "message": "This looks like a duplicate entry."}

    return {
        "status": "stored",
        "entry_id": entry.id,
        "entry_type": entry.entry_type.value,
        "title": entry.title,
        "needs_enrichment": entry.entry_type.value in ("movie", "game"),
    }


def sync_letterboxd(username: Optional[str] = None) -> dict:
    """
    Pull the full Letterboxd diary and store any new entries.
    Safe to call repeatedly — duplicates are skipped via the
    (source, source_id) unique constraint, caught in store_entry().
    """
    try:
        raw_entries = fetch_letterboxd_entries(username)
    except (ConnectionError, ValueError) as e:
        logger.error("Letterboxd sync failed: %s", e)
        return {"status": "error", "message": str(e), "stored": 0, "skipped": 0}

    stored, skipped = 0, 0
    stored_ids = []

    for raw in raw_entries:
        entry = store_entry(raw)
        if entry:
            stored += 1
            stored_ids.append(entry.id)
        else:
            skipped += 1

    return {
        "status": "ok",
        "source": "letterboxd",
        "fetched": len(raw_entries),
        "stored": stored,
        "skipped": skipped,
        "new_entry_ids": stored_ids,  # hand off to enrichment agent
    }


def sync_goodreads(csv_path: Optional[str] = None) -> dict:
    """Bulk import a Goodreads export CSV. Same dedup behaviour as Letterboxd."""
    try:
        raw_entries = import_goodreads_csv(csv_path)
    except FileNotFoundError as e:
        logger.error("Goodreads import failed: %s", e)
        return {"status": "error", "message": str(e), "stored": 0, "skipped": 0}

    stored, skipped = 0, 0
    for raw in raw_entries:
        entry = store_entry(raw)
        stored += 1 if entry else 0
        skipped += 0 if entry else 1

    return {
        "status": "ok",
        "source": "goodreads",
        "fetched": len(raw_entries),
        "stored": stored,
        "skipped": skipped,
    }


def sync_all_sources() -> list[dict]:
    """Run every configured sync in sequence. Used by the scheduler."""
    results = []
    for sync_fn in (sync_letterboxd, sync_goodreads):
        try:
            results.append(sync_fn())
        except Exception as e:  # noqa: BLE001 — top-level guard for the scheduler
            logger.exception("Sync failed unexpectedly")
            results.append({"status": "error", "message": str(e)})
    return results
