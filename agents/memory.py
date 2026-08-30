"""
agents/memory.py
Handles all reads/writes to the dual-store memory system: SQLite for
structured facts, ChromaDB for semantic embeddings. Every other agent
goes through this module rather than touching the DB layer directly.
"""

from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import Entry, EntryMetadata, EntryType, EntrySource, SessionLocal
from db.vector_store import vector_store


# ── Write ──────────────────────────────────────────────────────────────────

def store_entry(entry_data: dict, db: Optional[Session] = None) -> Optional[Entry]:
    """
    Persist a parsed/enriched entry to SQLite, then embed it into ChromaDB.

    entry_data is the dict shape produced by tools/parsers.py,
    tools/letterboxd.py, or tools/igdb.py — all share the same keys.

    Returns the created Entry, or None if it was a duplicate (safely skipped).
    """
    owns_session = db is None
    db = db or SessionLocal()

    try:
        entry = Entry(
            entry_type=EntryType(entry_data["entry_type"]),
            title=entry_data["title"],
            raw_text=entry_data.get("raw_text"),
            event_date=entry_data.get("event_date"),
            rating=entry_data.get("rating"),
            source=EntrySource(entry_data.get("source", "manual")),
            source_id=entry_data.get("source_id"),
            mood_score=entry_data.get("mood_score"),
            tags=entry_data.get("tags", []),
        )
        db.add(entry)
        db.flush()  # assigns entry.id without committing yet

        # Embed into ChromaDB using the entry's own id as the doc id
        entry.embedding_id = entry.id
        vector_store.upsert(
            embedding_id=entry.id,
            document=entry.to_chroma_document(),
            metadata=entry.to_chroma_metadata(),
        )

        db.commit()
        db.refresh(entry)
        return entry

    except IntegrityError:
        # Duplicate (source, source_id) — already imported, skip quietly.
        db.rollback()
        return None
    finally:
        if owns_session:
            db.close()


def attach_metadata(entry_id: str, meta: dict, db: Optional[Session] = None) -> None:
    """
    Attach enrichment metadata (from TMDB/IGDB) to an existing entry.
    Called by the enrichment agent after store_entry().
    """
    owns_session = db is None
    db = db or SessionLocal()

    try:
        existing = db.query(EntryMetadata).filter_by(entry_id=entry_id).first()
        if existing:
            for key in ("external_id", "poster_url", "genres", "creator",
                        "release_year", "duration_mins", "extra"):
                if key in meta:
                    setattr(existing, key, meta[key])
        else:
            db.add(EntryMetadata(entry_id=entry_id, **meta))
        db.commit()
    finally:
        if owns_session:
            db.close()


def update_entry_enrichment(entry_id: str, tags: list[str], mood_score: float,
                             db: Optional[Session] = None) -> None:
    """
    Update an entry's tags/mood after LLM enrichment, and re-sync the
    ChromaDB embedding since the document text has changed.
    """
    owns_session = db is None
    db = db or SessionLocal()

    try:
        entry = db.query(Entry).filter_by(id=entry_id).first()
        if not entry:
            return
        entry.tags = tags
        entry.mood_score = mood_score
        db.commit()
        db.refresh(entry)

        # Re-embed since tags changed the document text
        vector_store.upsert(
            embedding_id=entry.id,
            document=entry.to_chroma_document(),
            metadata=entry.to_chroma_metadata(),
        )
    finally:
        if owns_session:
            db.close()


# ── Read ───────────────────────────────────────────────────────────────────

def get_entries(
    entry_type: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 50,
    db: Optional[Session] = None,
) -> list[Entry]:
    """Structured query — use for counts, filters, date ranges."""
    owns_session = db is None
    db = db or SessionLocal()

    try:
        q = db.query(Entry)
        if entry_type:
            q = q.filter(Entry.entry_type == EntryType(entry_type))
        if start_date:
            q = q.filter(Entry.event_date >= start_date)
        if end_date:
            q = q.filter(Entry.event_date <= end_date)
        return q.order_by(Entry.event_date.desc()).limit(limit).all()
    finally:
        if owns_session:
            db.close()


def semantic_search(query: str, n_results: int = 5, entry_type: Optional[str] = None) -> list[dict]:
    """Semantic query — use for 'find me something like X' style questions."""
    where = {"entry_type": entry_type} if entry_type else None
    return vector_store.search(query, n_results=n_results, where=where)


def get_week_stats(week_start: date, db: Optional[Session] = None) -> dict:
    """
    Aggregate stats for the reflection/digest agent: counts by type,
    average mood, most common tags, for the 7-day window starting week_start.
    """
    owns_session = db is None
    db = db or SessionLocal()

    try:
        week_end = week_start + timedelta(days=7)
        entries = (
            db.query(Entry)
            .filter(Entry.event_date >= week_start, Entry.event_date < week_end)
            .all()
        )

        counts_by_type: dict[str, int] = {}
        mood_scores = []
        tag_counts: dict[str, int] = {}

        for e in entries:
            t = e.entry_type.value
            counts_by_type[t] = counts_by_type.get(t, 0) + 1
            if e.mood_score is not None:
                mood_scores.append(e.mood_score)
            for tag in (e.tags or []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        top_tags = sorted(tag_counts.items(), key=lambda kv: -kv[1])[:5]

        return {
            "total_entries": len(entries),
            "counts_by_type": counts_by_type,
            "avg_mood": sum(mood_scores) / len(mood_scores) if mood_scores else None,
            "top_tags": [t for t, _ in top_tags],
            "entries": entries,
        }
    finally:
        if owns_session:
            db.close()
