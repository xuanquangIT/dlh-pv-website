"""Tests for v2 execute_sql safety guards.

These are critical — execute_sql is the LLM's escape hatch for novel
queries. A weak validator = SQL injection / DDL / data loss.
"""
from __future__ import annotations

import unittest

from app.services.solar_ai_chat.primitives import (
    DEFAULT_MAX_ROWS,
    HARD_MAX_ROWS,
    execute_sql,
    validate_sql,
)


class ValidateSqlTests(unittest.TestCase):
    """Pure validation — no executor."""

    def test_simple_select_is_safe(self):
        result = validate_sql("SELECT facility_id FROM pv.gold.dim_facility")
        self.assertTrue(result.safe)
        self.assertEqual(result.violations, [])

    def test_with_cte_is_safe(self):
        result = validate_sql(
            "WITH t AS (SELECT facility_id FROM pv.gold.dim_facility) SELECT * FROM t"
        )
        self.assertTrue(result.safe)

    def test_values_literal_blocked(self):
        """Regression: model hallucinated facilities by writing
        SELECT * FROM (VALUES (...)) AS t — must be rejected because no
        real lakehouse table is referenced."""
        result = validate_sql(
            "SELECT * FROM (VALUES (1,'Desert Sun Mega',150,33.45,-114.71)) "
            "AS t(id, name, mw, lat, lng)"
        )
        self.assertFalse(result.safe)
        self.assertTrue(any("lakehouse" in v.lower() for v in result.violations))

    def test_select_constant_blocked(self):
        """`SELECT 1` doesn't read from any real table."""
        result = validate_sql("SELECT 1")
        self.assertFalse(result.safe)

    def test_insert_is_blocked(self):
        result = validate_sql("INSERT INTO foo VALUES (1)")
        self.assertFalse(result.safe)
        self.assertTrue(any("INSERT" in v for v in result.violations))

    def test_update_is_blocked(self):
        result = validate_sql("UPDATE foo SET x = 1")
        self.assertFalse(result.safe)

    def test_delete_is_blocked(self):
        result = validate_sql("DELETE FROM foo WHERE x = 1")
        self.assertFalse(result.safe)

    def test_drop_table_is_blocked(self):
        result = validate_sql("DROP TABLE foo")
        self.assertFalse(result.safe)

    def test_drop_in_subquery_is_blocked(self):
        # Even cleverly hidden in a subquery
        result = validate_sql("SELECT 1 FROM (DROP TABLE foo) x")
        self.assertFalse(result.safe)

    def test_truncate_is_blocked(self):
        result = validate_sql("TRUNCATE TABLE foo")
        self.assertFalse(result.safe)

    def test_alter_is_blocked(self):
        result = validate_sql("ALTER TABLE foo ADD COLUMN x INT")
        self.assertFalse(result.safe)

    def test_create_is_blocked(self):
        result = validate_sql("CREATE TABLE foo (x INT)")
        self.assertFalse(result.safe)

    def test_grant_is_blocked(self):
        result = validate_sql("GRANT SELECT ON foo TO PUBLIC")
        self.assertFalse(result.safe)

    def test_stacked_queries_blocked(self):
        result = validate_sql("SELECT 1; DROP TABLE foo")
        self.assertFalse(result.safe)
        self.assertTrue(any("Stacked" in v for v in result.violations))

    def test_information_schema_blocked(self):
        result = validate_sql("SELECT * FROM information_schema.tables")
        self.assertFalse(result.safe)

    def test_pg_catalog_blocked(self):
        result = validate_sql("SELECT * FROM pg_catalog.pg_tables")
        self.assertFalse(result.safe)

    def test_system_catalog_blocked(self):
        result = validate_sql("SELECT * FROM system.access_audit")
        self.assertFalse(result.safe)

    def test_auto_limit_appended_when_missing(self):
        result = validate_sql("SELECT * FROM pv.gold.dim_facility")
        self.assertTrue(result.safe)
        self.assertTrue(result.auto_limit_applied)
        self.assertIn("LIMIT", result.normalized_sql.upper())

    def test_existing_limit_respected(self):
        result = validate_sql("SELECT * FROM pv.gold.dim_facility LIMIT 5")
        self.assertTrue(result.safe)
        self.assertFalse(result.auto_limit_applied)

    def test_max_rows_capped_at_hard_max(self):
        result = validate_sql(
            "SELECT facility_id FROM pv.gold.dim_facility",
            max_rows=HARD_MAX_ROWS + 999_999,
        )
        # validate_sql doesn't clamp — execute_sql does. Just confirm no crash.
        self.assertTrue(result.safe)

    def test_empty_sql_violates(self):
        result = validate_sql("")
        self.assertFalse(result.safe)

    def test_lowercase_select_accepted(self):
        result = validate_sql("select * from pv.gold.dim_facility")
        self.assertTrue(result.safe)

    def test_lowercase_insert_blocked(self):
        result = validate_sql("insert into foo values (1)")
        self.assertFalse(result.safe)


class ExecuteSqlTests(unittest.TestCase):
    """End-to-end primitive behaviour with a fake executor."""

    def test_blocked_sql_returns_error_dict(self):
        result = execute_sql(
            sql="DROP TABLE pv.gold.dim_facility",
            sql_executor=lambda s: [{"x": 1}],
        )
        self.assertIn("error", result)
        self.assertIn("violations", result)
        self.assertEqual(result.get("rows"), None)

    def test_executor_called_with_normalized_sql(self):
        captured: list[str] = []

        def _fake(sql: str):
            captured.append(sql)
            return [{"facility_name": "Avonlie", "capacity_mw": 250}]

        result = execute_sql(
            sql="SELECT facility_name, capacity_mw FROM pv.gold.dim_facility",
            sql_executor=_fake,
        )
        self.assertEqual(result["row_count"], 1)
        self.assertIn("LIMIT", captured[0].upper())   # auto-limit applied
        self.assertEqual(result["columns"], ["facility_name", "capacity_mw"])
        self.assertTrue(result["auto_limit_applied"])

    def test_executor_exception_returns_error_with_guidance(self):
        def _broken(sql: str):
            raise RuntimeError("warehouse offline")

        result = execute_sql(
            sql="SELECT facility_id FROM pv.gold.dim_facility",
            sql_executor=_broken,
        )
        self.assertIn("error", result)
        self.assertIn("warehouse offline", result["error"])
        self.assertIn("guidance", result)

    def test_no_executor_returns_dry_run(self):
        result = execute_sql(sql="SELECT facility_id FROM pv.gold.dim_facility", sql_executor=None)
        self.assertIn("error", result)
        self.assertIn("executed_sql", result)

    def test_max_rows_clamped_to_hard_max(self):
        captured: list[str] = []

        def _fake(sql: str):
            captured.append(sql)
            return []

        execute_sql(
            sql="SELECT facility_id FROM pv.gold.dim_facility",
            max_rows=HARD_MAX_ROWS + 1_000_000,
            sql_executor=_fake,
        )
        # The injected LIMIT must be <= HARD_MAX_ROWS
        import re
        m = re.search(r"LIMIT\s+(\d+)", captured[0], re.IGNORECASE)
        self.assertIsNotNone(m)
        self.assertLessEqual(int(m.group(1)), HARD_MAX_ROWS)


if __name__ == "__main__":
    unittest.main()
