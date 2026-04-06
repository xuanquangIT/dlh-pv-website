"""Unit tests for BaseRepository SQL rewriting and table mapping."""
import pytest

from app.core.settings import SolarChatSettings
from app.repositories.solar_ai_chat.base_repository import BaseRepository


def _make_repo(catalog: str = "iceberg") -> BaseRepository:
    """Create a BaseRepository with specified catalog."""
    settings = SolarChatSettings()
    settings.trino_catalog = catalog
    return BaseRepository(settings=settings)


class TestRewriteSqlForIceberg:
    """Test suite for _rewrite_sql_for_iceberg() method."""

    def test_silver_clean_facility_master_table_rewrite(self) -> None:
        """Test rewrite of lh_silver_clean_facility_master table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT * FROM lh_silver_clean_facility_master LIMIT 1"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.silver.clean_facility_master" in rewritten
        assert "lh_silver_clean_facility_master" not in rewritten

    def test_silver_clean_hourly_energy_table_rewrite(self) -> None:
        """Test rewrite of lh_silver_clean_hourly_energy table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT * FROM lh_silver_clean_hourly_energy WHERE date_hour > '2026-01-01'"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.silver.clean_hourly_energy" in rewritten
        assert "lh_silver_clean_hourly_energy" not in rewritten

    def test_silver_clean_hourly_weather_table_rewrite(self) -> None:
        """Test rewrite of lh_silver_clean_hourly_weather table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT * FROM lh_silver_clean_hourly_weather"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.silver.clean_hourly_weather" in rewritten
        assert "lh_silver_clean_hourly_weather" not in rewritten

    def test_silver_clean_hourly_air_quality_table_rewrite(self) -> None:
        """Test rewrite of lh_silver_clean_hourly_air_quality table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT aqi_value FROM lh_silver_clean_hourly_air_quality"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.silver.clean_hourly_air_quality" in rewritten
        assert "lh_silver_clean_hourly_air_quality" not in rewritten

    def test_gold_dim_date_table_rewrite(self) -> None:
        """Test rewrite of lh_gold_dim_date table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT * FROM lh_gold_dim_date WHERE full_date >= '2026-01-01'"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.gold.dim_date" in rewritten
        assert "lh_gold_dim_date" not in rewritten

    def test_gold_dim_time_table_rewrite(self) -> None:
        """Test rewrite of lh_gold_dim_time table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT time_key FROM lh_gold_dim_time ORDER BY time_key"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.gold.dim_time" in rewritten
        assert "lh_gold_dim_time" not in rewritten

    def test_gold_dim_aqi_category_table_rewrite(self) -> None:
        """Test rewrite of lh_gold_dim_aqi_category table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT aqi_category_key, category_name FROM lh_gold_dim_aqi_category"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.gold.dim_aqi_category" in rewritten
        assert "lh_gold_dim_aqi_category" not in rewritten

    def test_gold_fact_solar_environmental_table_rewrite(self) -> None:
        """Test rewrite of lh_gold_fact_solar_environmental table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT energy_mwh FROM lh_gold_fact_solar_environmental"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.gold.fact_solar_environmental" in rewritten
        assert "lh_gold_fact_solar_environmental" not in rewritten

    def test_gold_dim_facility_table_rewrite(self) -> None:
        """Test rewrite of lh_gold_dim_facility table."""
        repo = _make_repo(catalog="iceberg")
        sql = "SELECT facility_key, facility_code FROM lh_gold_dim_facility"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.gold.dim_facility" in rewritten
        assert "lh_gold_dim_facility" not in rewritten

    def test_complex_join_with_multiple_tables(self) -> None:
        """Test rewrite of complex JOIN query across multiple tables."""
        repo = _make_repo(catalog="iceberg")
        sql = """
            SELECT 
                d.full_date,
                t.time_key,
                f.energy_mwh,
                w.temperature_2m,
                a.aqi_value,
                fac.facility_code
            FROM lh_gold_dim_date d
            JOIN lh_gold_dim_time t ON d.date_key = t.time_key
            JOIN lh_gold_fact_solar_environmental f ON f.date_key = d.date_key
            JOIN lh_silver_clean_hourly_weather w ON w.facility_code = fac.facility_code
            JOIN lh_silver_clean_hourly_air_quality a ON a.facility_code = fac.facility_code
            JOIN lh_gold_dim_facility fac ON fac.facility_key = f.facility_key
            WHERE d.full_date >= '2026-01-01'
        """
        rewritten = repo._rewrite_sql_for_iceberg(sql)

        # Verify all legacy table names are rewritten
        assert "lh_gold_dim_date" not in rewritten
        assert "lh_gold_dim_time" not in rewritten
        assert "lh_gold_fact_solar_environmental" not in rewritten
        assert "lh_silver_clean_hourly_weather" not in rewritten
        assert "lh_silver_clean_hourly_air_quality" not in rewritten
        assert "lh_gold_dim_facility" not in rewritten

        # Verify all table names are qualified with iceberg catalog
        assert "iceberg.gold.dim_date" in rewritten
        assert "iceberg.gold.dim_time" in rewritten
        assert "iceberg.gold.fact_solar_environmental" in rewritten
        assert "iceberg.silver.clean_hourly_weather" in rewritten
        assert "iceberg.silver.clean_hourly_air_quality" in rewritten
        assert "iceberg.gold.dim_facility" in rewritten

    def test_no_rewrite_for_non_iceberg_catalog(self) -> None:
        """Test that SQL is not rewritten when catalog is not iceberg.
        
        Note: 'postgresql' is automatically converted to 'iceberg' by _resolve_trino_catalog(),
        so we test with a truly different catalog name.
        """
        repo = _make_repo(catalog="some_other_catalog")
        sql = "SELECT * FROM lh_silver_clean_hourly_energy"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        # Should return unchanged because catalog is not iceberg
        assert rewritten == sql
        assert "lh_silver_clean_hourly_energy" in rewritten

    def test_table_names_as_aliases_not_rewritten(self) -> None:
        """Test that table names used as aliases in alias clauses are not rewritten when not table references."""
        repo = _make_repo(catalog="iceberg")
        # The table reference should be rewritten, but alias references shouldn't cause issues
        sql = "SELECT e.energy_mwh FROM lh_silver_clean_hourly_energy e"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        assert "iceberg.silver.clean_hourly_energy e" in rewritten
        assert "lh_silver_clean_hourly_energy" not in rewritten

    def test_table_name_with_word_boundaries(self) -> None:
        """Test that table names match word boundaries correctly (no partial matches)."""
        repo = _make_repo(catalog="iceberg")
        # Verify that similar but different names don't interfere
        sql = "SELECT * FROM lh_silver_clean_hourly_energy LIMIT 10"
        rewritten = repo._rewrite_sql_for_iceberg(sql)
        
        # The exact table name should be rewritten
        assert "iceberg.silver.clean_hourly_energy" in rewritten
        # Verify no double-rewriting or partial matches
        assert rewritten.count("iceberg.silver.clean_hourly_energy") == 1

    def test_all_table_mappings_covered(self) -> None:
        """Test that all 9 expected table mappings are present in _ICEBERG_TABLE_MAP."""
        repo = _make_repo(catalog="iceberg")
        
        expected_mappings = {
            "lh_silver_clean_facility_master",
            "lh_silver_clean_hourly_energy",
            "lh_silver_clean_hourly_weather",
            "lh_silver_clean_hourly_air_quality",
            "lh_gold_dim_date",
            "lh_gold_dim_time",
            "lh_gold_dim_aqi_category",
            "lh_gold_fact_solar_environmental",
            "lh_gold_dim_facility",
        }
        
        actual_mappings = set(repo._ICEBERG_TABLE_MAP.keys())
        
        assert actual_mappings == expected_mappings, (
            f"Table mapping mismatch. Missing: {expected_mappings - actual_mappings}. "
            f"Extra: {actual_mappings - expected_mappings}"
        )

    def test_all_table_mappings_have_correct_schema_prefix(self) -> None:
        """Test that all mappings have correct 'silver.' or 'gold.' prefix."""
        repo = _make_repo(catalog="iceberg")
        
        for legacy_name, iceberg_name in repo._ICEBERG_TABLE_MAP.items():
            assert iceberg_name.startswith(("silver.", "gold.")), (
                f"Mapping '{legacy_name}' -> '{iceberg_name}' must start with 'silver.' or 'gold.'"
            )
            
            # Verify schema is correct based on table name prefix
            if legacy_name.startswith("lh_silver"):
                assert iceberg_name.startswith("silver."), (
                    f"Silver table '{legacy_name}' must map to 'silver.' schema, got '{iceberg_name}'"
                )
            elif legacy_name.startswith("lh_gold"):
                assert iceberg_name.startswith("gold."), (
                    f"Gold table '{legacy_name}' must map to 'gold.' schema, got '{iceberg_name}'"
                )
