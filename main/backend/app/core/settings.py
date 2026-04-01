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
        default=3.2,
        alias="SOLAR_CHAT_REQUEST_TIMEOUT_SECONDS",
    )
    data_root: str | None = Field(default=None, alias="SOLAR_CHAT_DATA_ROOT")

    trino_host: str = Field(default="localhost", alias="TRINO_HOST")
    trino_port: int = Field(default=8081, alias="TRINO_PORT")
    trino_user: str = Field(default="trino", alias="TRINO_USER")
    trino_catalog: str = Field(default="postgresql", alias="TRINO_CATALOG")
    trino_schema: str = Field(default="public", alias="TRINO_SCHEMA")

    @property
    def resolved_data_root(self) -> Path:
        if self.data_root:
            return Path(self.data_root).resolve()

        repository_root = Path(__file__).resolve().parents[4]
        return repository_root / "main" / "sql"


@lru_cache(maxsize=1)
def get_solar_chat_settings() -> SolarChatSettings:
    return SolarChatSettings()
