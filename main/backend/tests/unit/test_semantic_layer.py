"""Tests for v2 semantic layer loader + RBAC filtering."""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.solar_ai_chat.semantic_loader import (
    invalidate_cache,
    load_semantic_layer,
)
from app.services.solar_ai_chat.primitives import (
    discover_schema,
    inspect_table,
    recall_metric,
    render_visualization,
)


_TEST_YAML = textwrap.dedent("""
version: 1
catalogs:
  pv:
    description: Test catalog
    schemas:
      gold:
        description: Test schema
        tables:
          dim_facility:
            description: "Facility master with lat/lng for map rendering"
            grain: [facility_id]
            columns:
              - {name: facility_id, type: string}
              - {name: facility_name, type: string}
              - {name: latitude, type: numeric}
              - {name: longitude, type: numeric}
            sample_questions: ["Where are facilities?"]
          mart_energy_daily:
            description: "Daily energy per facility"
            grain: [facility_id, date]
            columns:
              - {name: facility_id, type: string}
              - {name: total_energy_mwh, type: numeric}

metrics:
  facility_locations_map:
    description: "Toạ độ tất cả trạm cho map visualization"
    synonyms: ["bản đồ", "map", "ban do", "vị trí"]
    sample_questions:
      - "Cho tôi xem map các trạm"
      - "Where are the facilities"
    sql_template: "SELECT facility_name, latitude, longitude FROM pv.gold.dim_facility"
    parameters: []
    suggested_chart:
      chart_type: scatter_geo

  top_facilities_by_energy:
    description: "Top N trạm theo sản lượng điện"
    sql_template: |
      SELECT facility_id, SUM(total_energy_mwh) AS total
      FROM pv.gold.mart_energy_daily
      WHERE date >= CURRENT_DATE - INTERVAL '{window_days}' DAY
      GROUP BY facility_id
      ORDER BY total DESC
      LIMIT {n}
    parameters:
      - {name: window_days, type: int, default: 30}
      - {name: n, type: int, default: 5}
    suggested_chart:
      chart_type: bar

roles:
  admin:
    description: full
    allowed_tables: ["*"]
    allowed_metrics: ["*"]
  data_analyst:
    description: marts only
    allowed_tables:
      - pv.gold.mart_energy_daily
    allowed_metrics:
      - top_facilities_by_energy
""").lstrip()


class SemanticLayerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._path = Path(self._tmp.name) / "metrics.yaml"
        self._path.write_text(_TEST_YAML, encoding="utf-8")
        invalidate_cache()
        self.layer = load_semantic_layer(str(self._path))

    def tearDown(self):
        invalidate_cache()
        self._tmp.cleanup()

    def test_loads_tables_with_fqn(self):
        fqns = sorted(t.fqn for t in self.layer.tables)
        self.assertEqual(fqns, ["pv.gold.dim_facility", "pv.gold.mart_energy_daily"])

    def test_loads_metrics(self):
        names = sorted(m.name for m in self.layer.metrics)
        self.assertEqual(names, ["facility_locations_map", "top_facilities_by_energy"])

    def test_role_policy_admin_sees_all(self):
        tables = self.layer.tables_for_role("admin")
        self.assertEqual(len(tables), 2)

    def test_role_policy_analyst_filtered(self):
        tables = self.layer.tables_for_role("data_analyst")
        self.assertEqual([t.fqn for t in tables], ["pv.gold.mart_energy_daily"])

    def test_role_policy_unknown_role_empty(self):
        tables = self.layer.tables_for_role("nobody")
        self.assertEqual(tables, ())


class DiscoverSchemaTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._path = Path(self._tmp.name) / "metrics.yaml"
        self._path.write_text(_TEST_YAML, encoding="utf-8")
        invalidate_cache()
        self.layer = load_semantic_layer(str(self._path))

    def tearDown(self):
        invalidate_cache()
        self._tmp.cleanup()

    def test_admin_sees_all(self):
        result = discover_schema(role_id="admin", semantic_layer=self.layer)
        self.assertEqual(result["total"], 2)

    def test_analyst_sees_only_marts(self):
        result = discover_schema(role_id="data_analyst", semantic_layer=self.layer)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["tables"][0]["fqn"], "pv.gold.mart_energy_daily")

    def test_domain_filter_by_description(self):
        result = discover_schema(
            role_id="admin", domain="map", semantic_layer=self.layer,
        )
        # Only dim_facility mentions "map" in its description
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["tables"][0]["fqn"], "pv.gold.dim_facility")

    def test_unknown_role_returns_empty(self):
        result = discover_schema(role_id="nobody", semantic_layer=self.layer)
        self.assertEqual(result["total"], 0)


class InspectTableTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._path = Path(self._tmp.name) / "metrics.yaml"
        self._path.write_text(_TEST_YAML, encoding="utf-8")
        invalidate_cache()
        self.layer = load_semantic_layer(str(self._path))

    def tearDown(self):
        invalidate_cache()
        self._tmp.cleanup()

    def test_returns_columns_with_types(self):
        result = inspect_table(
            table_fqn="pv.gold.dim_facility",
            role_id="admin",
            sample_executor=None,
            semantic_layer=self.layer,
        )
        self.assertEqual(result["fqn"], "pv.gold.dim_facility")
        col_names = [c["name"] for c in result["columns"]]
        self.assertIn("latitude", col_names)
        self.assertIn("longitude", col_names)

    def test_role_blocked_table_returns_error(self):
        result = inspect_table(
            table_fqn="pv.gold.dim_facility",
            role_id="data_analyst",
            sample_executor=None,
            semantic_layer=self.layer,
        )
        self.assertIn("error", result)

    def test_unknown_table_returns_error_with_suggestions(self):
        result = inspect_table(
            table_fqn="pv.gold.nonexistent",
            role_id="admin",
            sample_executor=None,
            semantic_layer=self.layer,
        )
        self.assertIn("error", result)
        self.assertIn("available_tables", result)

    def test_sample_executor_invoked(self):
        captured: list[str] = []

        def _fake_executor(sql: str):
            captured.append(sql)
            return [{"facility_id": "WRSF1", "facility_name": "White Rock"}]

        result = inspect_table(
            table_fqn="pv.gold.dim_facility",
            role_id="admin",
            sample_executor=_fake_executor,
            semantic_layer=self.layer,
        )
        self.assertEqual(len(captured), 1)
        self.assertIn("LIMIT 3", captured[0])
        self.assertEqual(len(result["sample_rows"]), 1)

    def test_sample_executor_failure_does_not_crash(self):
        def _broken(sql: str):
            raise RuntimeError("offline")

        result = inspect_table(
            table_fqn="pv.gold.dim_facility",
            role_id="admin",
            sample_executor=_broken,
            semantic_layer=self.layer,
        )
        self.assertEqual(result["sample_rows"], [])
        self.assertIn("offline", result["sample_error"])


class RecallMetricTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._path = Path(self._tmp.name) / "metrics.yaml"
        self._path.write_text(_TEST_YAML, encoding="utf-8")
        invalidate_cache()
        self.layer = load_semantic_layer(str(self._path))

    def tearDown(self):
        invalidate_cache()
        self._tmp.cleanup()

    def test_map_query_returns_facility_locations_first(self):
        result = recall_metric(
            query="show me a map of facility locations",
            role_id="admin",
            semantic_layer=self.layer,
        )
        names = [m["name"] for m in result["matches"]]
        self.assertIn("facility_locations_map", names)
        # Map metric should be the top match given the query
        self.assertEqual(names[0], "facility_locations_map")

    def test_top_n_query_returns_ranking_metric(self):
        result = recall_metric(
            query="top facilities by energy production",
            role_id="admin",
            semantic_layer=self.layer,
        )
        names = [m["name"] for m in result["matches"]]
        self.assertIn("top_facilities_by_energy", names)

    def test_unrelated_query_returns_empty_matches(self):
        result = recall_metric(
            query="completely unrelated cooking recipe pizza",
            role_id="admin",
            semantic_layer=self.layer,
        )
        self.assertEqual(len(result["matches"]), 0)

    def test_vietnamese_synonym_matches_map_metric(self):
        # Pure VN phrase with no English keyword overlap with metric name
        result = recall_metric(
            query="cho tôi xem bản đồ",
            role_id="admin",
            semantic_layer=self.layer,
        )
        names = [m["name"] for m in result["matches"]]
        self.assertIn("facility_locations_map", names)
        self.assertEqual(names[0], "facility_locations_map")

    def test_recall_metric_returns_synonyms_in_payload(self):
        result = recall_metric(
            query="map locations",
            role_id="admin",
            semantic_layer=self.layer,
        )
        self.assertTrue(result["matches"])
        top = result["matches"][0]
        self.assertIn("synonyms", top)
        self.assertIn("sample_questions", top)
        self.assertIn("bản đồ", top["synonyms"])

    def test_role_filters_metric_palette(self):
        result = recall_metric(
            query="map locations",
            role_id="data_analyst",  # has no facility_locations_map access
            semantic_layer=self.layer,
        )
        names = [m["name"] for m in result["matches"]]
        self.assertNotIn("facility_locations_map", names)


class RenderVisualizationTests(unittest.TestCase):
    def test_bar_spec_with_data_injected(self):
        result = render_visualization(
            spec={"mark": "bar", "encoding": {"x": {"field": "name"}}},
            data=[{"name": "A", "value": 10}, {"name": "B", "value": 20}],
            title="Test",
        )
        self.assertEqual(result["format"], "vega-lite")
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["spec"]["data"]["values"][0]["name"], "A")

    def test_geo_spec_supported(self):
        result = render_visualization(
            spec={"mark": "geoshape"},
            data=[{"latitude": 0, "longitude": 0}],
        )
        self.assertNotIn("error", result)

    def test_unsupported_mark_returns_error(self):
        result = render_visualization(
            spec={"mark": "donut"},   # not in vega-lite primitives
            data=[],
        )
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
