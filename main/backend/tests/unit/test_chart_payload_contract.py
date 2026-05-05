"""Contract test: v2 render_visualization output must match what the
frontend chart_renderer.js (Vega-Lite path) expects.

Frontend reads (see main/frontend/static/js/components/chart_renderer.js):
    payload.format === "vega-lite"     -> dispatches to vega-embed
    payload.spec                       -> passed straight to vegaEmbed()
    payload.title                      -> rendered as chart header
    payload.spec.mark                  -> used to detect geo specs for layout

If this test breaks, either the primitive contract changed (update frontend)
or the frontend contract changed (update primitive). Do not skip.
"""
from __future__ import annotations

import unittest

from app.services.solar_ai_chat.primitives import render_visualization


class RenderVisualizationFrontendContractTests(unittest.TestCase):

    def test_payload_has_format_field(self):
        """Frontend dispatch hinges on payload.format === 'vega-lite'."""
        result = render_visualization(
            spec={"mark": "bar", "encoding": {"x": {"field": "name"}}},
            data=[{"name": "A", "value": 1}],
        )
        self.assertEqual(result["format"], "vega-lite")

    def test_payload_has_spec_field_with_data_injected(self):
        """vega-embed reads spec.data.values, so primitive must inject data
        into the spec — the frontend never re-merges."""
        result = render_visualization(
            spec={"mark": "bar"},
            data=[{"x": 1}, {"x": 2}],
        )
        self.assertIn("spec", result)
        self.assertIn("data", result["spec"])
        self.assertIn("values", result["spec"]["data"])
        self.assertEqual(result["spec"]["data"]["values"], [{"x": 1}, {"x": 2}])

    def test_geo_spec_mark_preserved(self):
        """Frontend detects geo via spec.mark === 'geoshape' to reserve more
        vertical space; this contract must hold."""
        result = render_visualization(
            spec={"mark": "geoshape", "projection": {"type": "mercator"}},
            data=[{"latitude": 0, "longitude": 0}],
        )
        self.assertEqual(result["spec"]["mark"], "geoshape")
        self.assertEqual(result["spec"]["projection"]["type"], "mercator")

    def test_title_propagates_into_payload(self):
        """Frontend renders payload.title in a header div above the plot."""
        result = render_visualization(
            spec={"mark": "line"},
            data=[],
            title="Top facilities",
        )
        self.assertEqual(result["title"], "Top facilities")

    def test_explicit_title_in_spec_wins_over_param(self):
        """If LLM puts title inside the spec, do not overwrite — Vega-Lite
        renders it natively. The 'title' parameter is a fallback only."""
        result = render_visualization(
            spec={"mark": "bar", "title": "Spec-level title"},
            data=[],
            title="Param-level title",
        )
        # The spec-level title is what vega-embed actually renders.
        self.assertEqual(result["spec"]["title"], "Spec-level title")

    def test_row_count_reported(self):
        """Useful for the chart-status pill in the UI."""
        result = render_visualization(
            spec={"mark": "bar"},
            data=[{"x": 1}, {"x": 2}, {"x": 3}],
        )
        self.assertEqual(result["row_count"], 3)

    def test_unsupported_mark_returns_error_not_payload(self):
        """Frontend has no fallback for unknown marks — primitive must
        block them with an error envelope so chat_service can degrade."""
        result = render_visualization(
            spec={"mark": "donut"},   # invalid Vega-Lite mark
            data=[],
        )
        self.assertIn("error", result)
        self.assertNotIn("format", result)


if __name__ == "__main__":
    unittest.main()
