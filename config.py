from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # --- LLM ---
    # openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    llm_provider: str = Field("anthropic", env="LLM_PROVIDER")
    llm_model: str = Field("claude-sonnet-4-20250514", env="LLM_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")

    # --- External APIs ---
    tmdb_api_key: str = Field("", env="TMDB_API_KEY")       # themoviedb.org
    igdb_client_id: str = Field("", env="IGDB_CLIENT_ID")   # Twitch dev console
    igdb_client_secret: str = Field("", env="IGDB_CLIENT_SECRET")

    # --- Letterboxd ---
    letterboxd_username: str = Field("", env="LETTERBOXD_USERNAME")

    # --- Goodreads ---
    # Export CSV from: goodreads.com/review/import
    goodreads_csv_path: str = Field("", env="GOODREADS_CSV_PATH")

    # --- Database ---
    database_url: str = Field(
        f"sqlite:///{BASE_DIR}/db/hobby_tracker.db",
        env="DATABASE_URL",
    )
    chroma_persist_dir: str = Field(
        str(BASE_DIR / "db" / "chroma"),
        env="CHROMA_PERSIST_DIR",
    )
    chroma_collection_name: str = "entries"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
