from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SolarChatSettings(BaseSettings):
    """Runtime settings for Solar AI Chat module."""

    model_config = SettingsConfigDict(
        env_file=(".env", "dev/config/.env", "main/docker/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # allow field names in addition to aliases in tests/code
    )

    gemini_api_key: str | None = Field(default=None, alias="SOLAR_CHAT_GEMINI_API_KEY")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        alias="SOLAR_CHAT_GEMINI_BASE_URL",
    )
    primary_model: str = Field(
        default="gemini-2.5-flash-lite",
        alias="SOLAR_CHAT_PRIMARY_MODEL",
    )
    fallback_model: str = Field(
        default="gemini-2.5-flash",
        alias="SOLAR_CHAT_FALLBACK_MODEL",
    )
    request_timeout_seconds: float = Field(
        default=15.0,
        alias="SOLAR_CHAT_REQUEST_TIMEOUT_SECONDS",
    )
    data_root: str | None = Field(default=None, alias="SOLAR_CHAT_DATA_ROOT")

    trino_host: str = Field(default="localhost", alias="TRINO_HOST")
    trino_port: int = Field(default=8081, alias="TRINO_PORT")
    trino_user: str = Field(default="trino", alias="TRINO_USER")
    trino_catalog: str = Field(default="postgresql", alias="TRINO_CATALOG")
    trino_schema: str = Field(default="public", alias="TRINO_SCHEMA")

    pg_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    pg_port: int = Field(default=5432, alias="POSTGRES_PORT")
    pg_database: str = Field(default="pvlakehouse", alias="POSTGRES_DB")
    pg_user: str = Field(default="pvlakehouse", alias="POSTGRES_USER")
    pg_password: str = Field(default="pvlakehouse", alias="POSTGRES_PASSWORD")

    embedding_model: str = Field(
        default="text-embedding-004",  # 3072-dim model aligned with embedding_dimensions default
        alias="SOLAR_CHAT_EMBEDDING_MODEL",
    )
    embedding_dimensions: int = Field(
        default=3072,
        alias="SOLAR_CHAT_EMBEDDING_DIMENSIONS",
    )
    rag_chunk_size: int = Field(default=512, alias="SOLAR_CHAT_RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=64, alias="SOLAR_CHAT_RAG_CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="SOLAR_CHAT_RAG_TOP_K")

    @property
    def resolved_data_root(self) -> Path:
        if self.data_root:
            return Path(self.data_root).resolve()

        repository_root = Path(__file__).resolve().parents[4]
        return repository_root / "main" / "sql"


@lru_cache(maxsize=1)
def get_solar_chat_settings() -> SolarChatSettings:
    return SolarChatSettings()
