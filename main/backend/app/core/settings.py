import threading
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ENV_FILES = (
    str(PROJECT_ROOT / ".env"),
    str(PROJECT_ROOT / "main" / "docker" / ".env"),
)


class SolarChatSettings(BaseSettings):
    """Runtime settings for Solar AI Chat module."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_api_format: str = Field(
        default="gemini",
        validation_alias=AliasChoices(
            "SOLAR_CHAT_LLM_API_FORMAT",
            "SOLAR_CHAT_LLM_PROVIDER",
            "llm_api_format",
            "llm_provider",
        ),
    )
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SOLAR_CHAT_LLM_API_KEY",
            "SOLAR_CHAT_GEMINI_API_KEY",
            "llm_api_key",
            "gemini_api_key",
        ),
    )
    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SOLAR_CHAT_LLM_BASE_URL",
            "SOLAR_CHAT_GEMINI_BASE_URL",
            "llm_base_url",
            "gemini_base_url",
        ),
    )
    primary_model: str = Field(
        default="gemini-2.5-flash-lite",
        validation_alias=AliasChoices("SOLAR_CHAT_PRIMARY_MODEL", "primary_model"),
    )
    fallback_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("SOLAR_CHAT_FALLBACK_MODEL", "fallback_model"),
    )
    request_timeout_seconds: float = Field(
        default=90.0,
        validation_alias=AliasChoices("SOLAR_CHAT_REQUEST_TIMEOUT_SECONDS", "request_timeout_seconds"),
    )
    llm_default_max_output_tokens: int = Field(
        default=1800,
        validation_alias=AliasChoices(
            "SOLAR_CHAT_LLM_DEFAULT_MAX_OUTPUT_TOKENS",
            "SOLAR_AI_DEFAULT_MAX_OUTPUT_TOKENS",
            "llm_default_max_output_tokens",
        ),
    )
    llm_tool_call_max_output_tokens: int = Field(
        default=2000,
        validation_alias=AliasChoices(
            "SOLAR_CHAT_LLM_TOOL_CALL_MAX_OUTPUT_TOKENS",
            "SOLAR_AI_TOOL_CALL_MAX_OUTPUT_TOKENS",
            "llm_tool_call_max_output_tokens",
        ),
    )
    llm_reasoning_effort: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SOLAR_CHAT_REASONING_EFFORT",
            "llm_reasoning_effort",
        ),
    )
    llm_anthropic_version: str = Field(
        default="2023-06-01",
        alias="SOLAR_CHAT_ANTHROPIC_VERSION",
    )
    data_root: str | None = Field(default=None, alias="SOLAR_CHAT_DATA_ROOT")

    databricks_host: str | None = Field(default=None, alias="DATABRICKS_HOST")
    databricks_token: str | None = Field(default=None, alias="DATABRICKS_TOKEN")
    databricks_sql_http_path: str | None = Field(default=None, alias="DATABRICKS_SQL_HTTP_PATH")
    databricks_warehouse_id: str | None = Field(default=None, alias="DATABRICKS_WAREHOUSE_ID")
    use_separate_warehouse_for_solar_chat: bool = Field(
        default=False,
        alias="USE_SEPARATE_WAREHOUSE_FOR_SOLAR_CHAT",
    )
    solar_chat_databricks_host: str | None = Field(
        default=None, alias="SOLAR_CHAT_DATABRICKS_HOST"
    )
    solar_chat_databricks_token: str | None = Field(
        default=None, alias="SOLAR_CHAT_DATABRICKS_TOKEN"
    )
    solar_chat_databricks_sql_http_path: str | None = Field(
        default=None, alias="SOLAR_CHAT_DATABRICKS_SQL_HTTP_PATH"
    )
    solar_chat_databricks_warehouse_id: str | None = Field(
        default=None, alias="SOLAR_CHAT_DATABRICKS_WAREHOUSE_ID"
    )
    databricks_query_timeout_seconds: float = Field(
        default=12.0,
        validation_alias=AliasChoices(
            "SOLAR_CHAT_DATABRICKS_QUERY_TIMEOUT_SECONDS",
            "databricks_query_timeout_seconds",
        ),
    )
    allow_csv_fallback: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "SOLAR_CHAT_ALLOW_CSV_FALLBACK",
            "allow_csv_fallback",
        ),
    )
    uc_catalog: str = Field(default="pv", alias="UC_CATALOG")
    uc_app_catalog: str = Field(default="dlh-web", alias="UC_APP_CATALOG")
    uc_silver_schema: str = Field(default="silver", alias="UC_SILVER_SCHEMA")
    uc_gold_schema: str = Field(default="gold", alias="UC_GOLD_SCHEMA")
    uc_app_schema: str = Field(default="app", alias="UC_APP_SCHEMA")

    # Backward compatibility for existing env files that still include TRINO_* keys.
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
    pg_database_url: str | None = Field(default=None, alias="DATABASE_URL")
    pg_sslmode: str | None = Field(default=None, alias="POSTGRES_SSLMODE")
    pg_channel_binding: str | None = Field(default=None, alias="POSTGRES_CHANNEL_BINDING")

    @property
    def gemini_api_key(self) -> str | None:
        """Backward-compatible alias for legacy callsites/tests."""
        return self.llm_api_key

    @gemini_api_key.setter
    def gemini_api_key(self, value: str | None) -> None:
        self.llm_api_key = value

    @property
    def gemini_base_url(self) -> str | None:
        """Backward-compatible alias for legacy callsites/tests."""
        return self.llm_base_url

    @gemini_base_url.setter
    def gemini_base_url(self, value: str | None) -> None:
        self.llm_base_url = value

    @property
    def resolved_databricks_http_path(self) -> str | None:
        if self.databricks_sql_http_path:
            return self.databricks_sql_http_path.strip()

        if self.databricks_warehouse_id:
            warehouse_id = self.databricks_warehouse_id.strip()
            if warehouse_id:
                return f"/sql/1.0/warehouses/{warehouse_id}"

        return None

    @property
    def solar_chat_databricks_host_resolved(self) -> str | None:
        if self.use_separate_warehouse_for_solar_chat and (self.solar_chat_databricks_host or "").strip():
            return self.solar_chat_databricks_host.strip()
        return self.databricks_host

    @property
    def solar_chat_databricks_token_resolved(self) -> str | None:
        if self.use_separate_warehouse_for_solar_chat and (self.solar_chat_databricks_token or "").strip():
            return self.solar_chat_databricks_token.strip()
        return self.databricks_token

    @property
    def solar_chat_databricks_http_path_resolved(self) -> str | None:
        if self.use_separate_warehouse_for_solar_chat:
            if (self.solar_chat_databricks_sql_http_path or "").strip():
                return self.solar_chat_databricks_sql_http_path.strip()
            warehouse_id = (self.solar_chat_databricks_warehouse_id or "").strip()
            if warehouse_id:
                return f"/sql/1.0/warehouses/{warehouse_id}"
        return self.resolved_databricks_http_path

    @property
    def resolved_llm_base_url(self) -> str:
        if self.llm_base_url and self.llm_base_url.strip():
            return self.llm_base_url.strip()

        api_format = self.resolved_llm_api_format
        if api_format == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta"

        if api_format == "anthropic":
            return "https://api.anthropic.com/v1"

        return "https://api.openai.com/v1"

    @property
    def resolved_llm_api_format(self) -> str:
        """Pick wire format with priority: explicit format > cloud URL > model name.

        The explicit-format-wins rule is important for proxies (e.g. local
        Copilot relays) that serve Claude/Gemini-named models over the
        OpenAI wire protocol — naming "claude-haiku" should not flip the
        client to Anthropic format if the user already declared ``openai``.
        """
        normalized = self.llm_api_format.strip().lower()

        # Local / openai-compatible aliases always speak OpenAI wire format.
        if normalized in {"local", "openai-compatible", "openai_compatible"}:
            return "openai"

        # Hard-evidence cloud URLs (highest confidence — overrides everything).
        base_url = (self.llm_base_url or "").strip().lower()
        if "generativelanguage.googleapis.com" in base_url:
            return "gemini"
        if "api.anthropic.com" in base_url:
            return "anthropic"
        if "api.openai.com" in base_url or "api.groq.com" in base_url or "/openai" in base_url:
            return "openai"

        # Trust explicit user-declared format next. Critical for proxies that
        # expose Claude/Gemini model names over OpenAI wire protocol.
        if normalized in {"openai", "chatgpt", "groq"}:
            return "openai"
        if normalized in {"anthropic", "claude"}:
            return "anthropic"
        if normalized == "gemini":
            return "gemini"

        # Last-resort heuristic: model name prefix.
        primary_model = self.primary_model.strip().lower()
        if primary_model.startswith("gemini"):
            return "gemini"
        if primary_model.startswith("claude"):
            return "anthropic"

        return "openai"

    @property
    def resolved_data_root(self) -> Path:
        if self.data_root:
            return Path(self.data_root).resolve()

        repository_root = Path(__file__).resolve().parents[4]
        return repository_root / "main" / "sql"


@lru_cache(maxsize=1)
def get_solar_chat_settings() -> SolarChatSettings:
    return SolarChatSettings()

class PowerBISettings(BaseSettings):
    """Runtime settings for Power BI Embedded module."""

    model_config = SettingsConfigDict(
        env_file=(
            *DEFAULT_ENV_FILES,
            str(PROJECT_ROOT / "dev" / "config" / ".env"),
            str(PROJECT_ROOT / "stg" / "config" / ".env"),
            str(PROJECT_ROOT / "prod" / "config" / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tenant_id: str | None = Field(default=None, alias="POWERBI_TENANT_ID")
    client_id: str | None = Field(default=None, alias="POWERBI_CLIENT_ID")
    client_secret: str | None = Field(default=None, alias="POWERBI_CLIENT_SECRET")
    workspace_id: str | None = Field(default=None, alias="POWERBI_WORKSPACE_ID")
    report_id: str | None = Field(default=None, alias="POWERBI_REPORT_ID")

@lru_cache(maxsize=1)
def get_powerbi_settings() -> PowerBISettings:
    return PowerBISettings()


class DatabaseSettings(BaseSettings):
    """Runtime settings for Database module."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pg_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    pg_port: int = Field(default=5432, alias="POSTGRES_PORT")
    pg_database: str = Field(default="pvlakehouse", alias="POSTGRES_DB")
    pg_user: str = Field(default="pvlakehouse", alias="POSTGRES_USER")
    pg_password: str = Field(default="pvlakehouse", alias="POSTGRES_PASSWORD")
    pg_database_url: str | None = Field(default=None, alias="DATABASE_URL")
    pg_sslmode: str | None = Field(default=None, alias="POSTGRES_SSLMODE")
    pg_channel_binding: str | None = Field(default=None, alias="POSTGRES_CHANNEL_BINDING")

    @property
    def database_url(self) -> str:
        if self.pg_database_url:
            return self.pg_database_url.strip()

        url = f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        query_params: list[str] = []
        if self.pg_sslmode:
            query_params.append(f"sslmode={self.pg_sslmode.strip()}")
        if self.pg_channel_binding:
            query_params.append(f"channel_binding={self.pg_channel_binding.strip()}")
        if query_params:
            return f"{url}?{'&'.join(query_params)}"
        return url


class AuthSettings(BaseSettings):
    """Runtime settings for Authentication module."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    secret_key: str = Field(alias="AUTH_SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="AUTH_ALGORITHM")
    access_token_expire_minutes: int = Field(default=1440, alias="AUTH_ACCESS_TOKEN_EXPIRE_MINUTES")
    cookie_name: str = Field(default="pv_access_token", alias="AUTH_COOKIE_NAME")
    cookie_secure: bool = Field(default=False, alias="AUTH_COOKIE_SECURE")


_db_settings_instance = None
_db_settings_lock = threading.Lock()

def get_db_settings() -> DatabaseSettings:
    global _db_settings_instance
    with _db_settings_lock:
        if _db_settings_instance is None:
            _db_settings_instance = DatabaseSettings()
        return _db_settings_instance

_auth_settings_instance = None
_auth_settings_lock = threading.Lock()

def get_auth_settings() -> AuthSettings:
    global _auth_settings_instance
    with _auth_settings_lock:
        if _auth_settings_instance is None:
            _auth_settings_instance = AuthSettings()
        return _auth_settings_instance
