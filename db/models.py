"""
db/models.py
SQLAlchemy ORM models. Matches the schema we designed.
Run `python -m db.models` once to create all tables.
"""

import uuid
from datetime import datetime, date
from enum import Enum as PyEnum

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Float,
    Integer,
    Date,
    DateTime,
    JSON,
    ForeignKey,
    Enum,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config import settings

Base = declarative_base()


# ── Enums ──────────────────────────────────────────────────────────────────

class EntryType(str, PyEnum):
    movie = "movie"
    book = "book"
    game = "game"
    social = "social"


class EntrySource(str, PyEnum):
    manual = "manual"
    letterboxd = "letterboxd"
    goodreads = "goodreads"
    igdb = "igdb"


# ── Models ─────────────────────────────────────────────────────────────────

class Entry(Base):
    """One row per logged life event."""
    __tablename__ = "entries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    entry_type = Column(Enum(EntryType), nullable=False, index=True)
    title = Column(Text, nullable=False)
    raw_text = Column(Text, nullable=True)        # original freeform journal input
    logged_at = Column(DateTime, default=datetime.utcnow, index=True)
    event_date = Column(Date, nullable=True, index=True)  # when it actually happened
    rating = Column(Float, nullable=True)          # normalised 0–10
    source = Column(Enum(EntrySource), default=EntrySource.manual)
    source_id = Column(String(256), nullable=True) # external ID for dedup
    mood_score = Column(Float, nullable=True)      # LLM sentiment –1 to +1
    tags = Column(JSON, default=list)              # ["horror", "slow-burn"]
    embedding_id = Column(String(256), nullable=True)  # ChromaDB doc ID

    metadata_ = relationship(
        "EntryMetadata", back_populates="entry", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Prevent duplicate imports from the same external source
        UniqueConstraint("source", "source_id", name="uq_source_entry"),
        Index("ix_entries_event_date_type", "event_date", "entry_type"),
    )

    def to_chroma_document(self) -> str:
        """Produce the text we embed into ChromaDB."""
        parts = [self.title]
        if self.raw_text:
            parts.append(self.raw_text)
        if self.tags:
            parts.append(" ".join(self.tags))
        return " | ".join(parts)

    def to_chroma_metadata(self) -> dict:
        """Filterable metadata stored alongside the embedding."""
        return {
            "entry_type": self.entry_type.value if self.entry_type else None,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "rating": self.rating,
            "source": self.source.value if self.source else None,
        }

    def __repr__(self):
        return f"<Entry {self.entry_type} '{self.title}' {self.event_date}>"


class EntryMetadata(Base):
    """API-fetched enrichment data. Joined on demand, never in hot paths."""
    __tablename__ = "entry_metadata"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    entry_id = Column(String(36), ForeignKey("entries.id"), nullable=False, unique=True)
    external_id = Column(String(256), nullable=True)   # TMDB / IGDB / OpenLibrary ID
    poster_url = Column(Text, nullable=True)
    genres = Column(JSON, default=list)
    creator = Column(Text, nullable=True)              # director / author / studio
    release_year = Column(Integer, nullable=True)
    duration_mins = Column(Integer, nullable=True)     # movies + games
    extra = Column(JSON, default=dict)                 # type-specific catch-all

    entry = relationship("Entry", back_populates="metadata_")

    def __repr__(self):
        return f"<EntryMetadata entry_id={self.entry_id}>"


class Digest(Base):
    """Weekly LLM-generated reflections."""
    __tablename__ = "digests"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    week_start = Column(Date, nullable=False, unique=True, index=True)  # Monday
    summary = Column(Text, nullable=False)    # LLM-generated prose
    stats = Column(JSON, default=dict)        # counts, avg mood, top tags
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Digest week={self.week_start}>"


# ── Session factory ────────────────────────────────────────────────────────

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # needed for SQLite + threads
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """Dependency-injection style session getter. Use as a context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created.")


if __name__ == "__main__":
    init_db()
