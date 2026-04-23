"""Extended coverage for SolarChatSettings / DatabaseSettings property resolvers."""
from __future__ import annotations

from app.core.settings import DatabaseSettings, SolarChatSettings


class TestGeminiApiKeyAlias:
    def test_getter_returns_llm_api_key(self) -> None:
        s = SolarChatSettings(SOLAR_CHAT_LLM_API_KEY="abc")
        assert s.gemini_api_key == "abc"

    def test_setter_updates_llm_api_key(self) -> None:
        s = SolarChatSettings()
        s.gemini_api_key = "xyz"
        assert s.llm_api_key == "xyz"


class TestGeminiBaseUrlAlias:
    def test_getter(self) -> None:
        s = SolarChatSettings(SOLAR_CHAT_LLM_BASE_URL="https://x.example")
        assert s.gemini_base_url == "https://x.example"

    def test_setter(self) -> None:
        s = SolarChatSettings()
        s.gemini_base_url = "https://y.example"
        assert s.llm_base_url == "https://y.example"


class TestResolvedDatabricksHttpPath:
    def test_explicit_http_path(self) -> None:
        s = SolarChatSettings(DATABRICKS_SQL_HTTP_PATH="/sql/1.0/warehouses/abc")
        assert s.resolved_databricks_http_path == "/sql/1.0/warehouses/abc"

    def test_derived_from_warehouse_id(self) -> None:
        s = SolarChatSettings(DATABRICKS_SQL_HTTP_PATH="", DATABRICKS_WAREHOUSE_ID="xyz123")
        assert s.resolved_databricks_http_path == "/sql/1.0/warehouses/xyz123"

    def test_none_when_unset(self) -> None:
        s = SolarChatSettings(DATABRICKS_SQL_HTTP_PATH="", DATABRICKS_WAREHOUSE_ID="")
        assert s.resolved_databricks_http_path is None


class TestResolvedLLMBaseUrl:
    def test_explicit_url(self) -> None:
        s = SolarChatSettings(SOLAR_CHAT_LLM_BASE_URL="https://custom.example/v1")
        assert s.resolved_llm_base_url == "https://custom.example/v1"

    def test_gemini_default(self) -> None:
        s = SolarChatSettings(
            SOLAR_CHAT_LLM_BASE_URL="",
            SOLAR_CHAT_LLM_API_FORMAT="gemini",
            SOLAR_CHAT_PRIMARY_MODEL="gemini-2.5-flash",
        )
        assert "generativelanguage" in s.resolved_llm_base_url

    def test_anthropic_default(self) -> None:
        s = SolarChatSettings(
            SOLAR_CHAT_LLM_BASE_URL="",
            SOLAR_CHAT_LLM_API_FORMAT="anthropic",
            SOLAR_CHAT_PRIMARY_MODEL="claude-opus-4-7",
        )
        assert "api.anthropic.com" in s.resolved_llm_base_url

    def test_openai_default(self) -> None:
        s = SolarChatSettings(
            SOLAR_CHAT_LLM_BASE_URL="",
            SOLAR_CHAT_LLM_API_FORMAT="openai",
            SOLAR_CHAT_PRIMARY_MODEL="gpt-4",
        )
        assert "api.openai.com" in s.resolved_llm_base_url


class TestResolvedLLMApiFormat:
    def test_local_normalized_to_openai(self) -> None:
        s = SolarChatSettings(SOLAR_CHAT_LLM_API_FORMAT="local", SOLAR_CHAT_LLM_BASE_URL="")
        assert s.resolved_llm_api_format == "openai"

    def test_claude_inferred_from_primary_model(self) -> None:
        s = SolarChatSettings(
            SOLAR_CHAT_LLM_API_FORMAT="unknown",
            SOLAR_CHAT_LLM_BASE_URL="",
            SOLAR_CHAT_PRIMARY_MODEL="claude-opus-4-7",
        )
        assert s.resolved_llm_api_format == "anthropic"

    def test_gemini_inferred_from_primary_model(self) -> None:
        s = SolarChatSettings(
            SOLAR_CHAT_LLM_API_FORMAT="unknown",
            SOLAR_CHAT_LLM_BASE_URL="",
            SOLAR_CHAT_PRIMARY_MODEL="gemini-2.5-flash",
        )
        assert s.resolved_llm_api_format == "gemini"

    def test_gemini_inferred_from_base_url(self) -> None:
        s = SolarChatSettings(
            SOLAR_CHAT_LLM_API_FORMAT="openai",
            SOLAR_CHAT_LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta",
            SOLAR_CHAT_PRIMARY_MODEL="custom",
        )
        assert s.resolved_llm_api_format == "gemini"

    def test_anthropic_from_base_url(self) -> None:
        s = SolarChatSettings(
            SOLAR_CHAT_LLM_API_FORMAT="openai",
            SOLAR_CHAT_LLM_BASE_URL="https://api.anthropic.com/v1",
            SOLAR_CHAT_PRIMARY_MODEL="custom",
        )
        assert s.resolved_llm_api_format == "anthropic"


class TestDatabaseSettings:
    def test_database_url_explicit_passthrough(self) -> None:
        s = DatabaseSettings(DATABASE_URL="postgresql://u:p@h:5432/d")
        assert s.database_url == "postgresql://u:p@h:5432/d"

    def test_database_url_built_from_components(self) -> None:
        s = DatabaseSettings(
            DATABASE_URL=None,
            POSTGRES_HOST="h", POSTGRES_PORT=5432,
            POSTGRES_DB="d", POSTGRES_USER="u", POSTGRES_PASSWORD="p",
            POSTGRES_SSLMODE=None, POSTGRES_CHANNEL_BINDING=None,
        )
        assert s.database_url == "postgresql://u:p@h:5432/d"

    def test_database_url_with_sslmode(self) -> None:
        s = DatabaseSettings(
            DATABASE_URL=None, POSTGRES_HOST="h", POSTGRES_PORT=5432,
            POSTGRES_DB="d", POSTGRES_USER="u", POSTGRES_PASSWORD="p",
            POSTGRES_SSLMODE="require",
        )
        assert "sslmode=require" in s.database_url

    def test_database_url_with_channel_binding(self) -> None:
        s = DatabaseSettings(
            DATABASE_URL=None, POSTGRES_HOST="h", POSTGRES_PORT=5432,
            POSTGRES_DB="d", POSTGRES_USER="u", POSTGRES_PASSWORD="p",
            POSTGRES_SSLMODE="require", POSTGRES_CHANNEL_BINDING="require",
        )
        assert "channel_binding=require" in s.database_url
