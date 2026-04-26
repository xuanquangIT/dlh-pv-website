/**
 * chart_renderer.js — Render a chart from a ChartPayload.
 *
 * Supports two payload shapes:
 *   1. v1 Plotly:    { plotly_spec: {data, layout}, chart_type, title, description }
 *   2. v2 Vega-Lite: { format: "vega-lite", spec: {...with .data.values...}, title, row_count }
 *
 * Dispatch is by `payload.format`: "vega-lite" -> vega-embed, otherwise Plotly.
 * Both libraries are loaded in platform_portal/base.html as deferred scripts.
 *
 * Exposes window.ChartRenderer.render(container, payload).
 */
(function (global) {
  "use strict";

  function escapeHtml(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function buildHeader(container, payload) {
    container.classList.add("solar-chart-wrap");
    container.innerHTML = "";
    if (payload.title) {
      var title = document.createElement("div");
      title.className = "solar-chart-title";
      title.textContent = payload.title;
      container.appendChild(title);
    }
    if (payload.description) {
      var desc = document.createElement("div");
      desc.className = "solar-chart-desc";
      desc.textContent = payload.description;
      container.appendChild(desc);
    }
  }

  function renderPlotly(container, payload) {
    var plotDiv = document.createElement("div");
    plotDiv.className = "solar-chart-plot";
    plotDiv.style.minHeight = payload.chart_type === "scatter_geo" ? "400px" : "320px";
    container.appendChild(plotDiv);

    if (!global.Plotly) {
      plotDiv.innerHTML = '<div class="solar-chart-fallback">Chart unavailable: Plotly.js not loaded.</div>';
      return;
    }

    var spec = payload.plotly_spec || {};
    var data = spec.data || [];
    var baseLayout = spec.layout || {};
    var cleanLayout = Object.assign({}, baseLayout);
    delete cleanLayout.title;
    var isGeo = payload.chart_type === "scatter_geo";
    var layout = Object.assign(
      {
        autosize: true,
        hovermode: isGeo ? "closest" : "x unified",
        height: isGeo ? 420 : 340,
        margin: isGeo ? { l: 0, r: 0, t: 10, b: 0 } : { l: 52, r: 18, t: 14, b: 48 },
        legend: { orientation: "h", y: -0.22, x: 0, xanchor: "left", font: { size: 11 } },
        font: { size: 11 }
      },
      cleanLayout
    );
    var config = { responsive: true, displayModeBar: false };
    try {
      global.Plotly.newPlot(plotDiv, data, layout, config).then(function () {
        try { global.Plotly.Plots.resize(plotDiv); } catch (_) {}
      });
      if (!global.__solarChartResizeBound) {
        global.__solarChartResizeBound = true;
        global.addEventListener("resize", function () {
          document.querySelectorAll(".solar-chart-plot").forEach(function (el) {
            try { global.Plotly.Plots.resize(el); } catch (_) {}
          });
        });
      }
    } catch (err) {
      plotDiv.innerHTML = '<div class="solar-chart-fallback">Chart render failed: ' +
        escapeHtml(err && err.message ? err.message : "unknown error") + "</div>";
    }
  }

  function renderVegaLite(container, payload) {
    var plotDiv = document.createElement("div");
    plotDiv.className = "solar-chart-plot solar-chart-vega";
    // Geo specs need more vertical room
    var spec = payload.spec || {};
    var mark = typeof spec.mark === "string" ? spec.mark : (spec.mark && spec.mark.type) || "";
    var isGeo = mark === "geoshape" || mark === "circle" && spec.projection;
    plotDiv.style.minHeight = isGeo ? "400px" : "320px";
    container.appendChild(plotDiv);

    if (!global.vegaEmbed) {
      plotDiv.innerHTML = '<div class="solar-chart-fallback">Chart unavailable: vega-embed not loaded.</div>';
      return;
    }
    if (!spec || typeof spec !== "object") {
      plotDiv.innerHTML = '<div class="solar-chart-fallback">Invalid Vega-Lite spec.</div>';
      return;
    }

    // Apply sensible defaults so LLM-emitted specs render at consistent size
    var fullSpec = Object.assign(
      {
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        width: "container",
        height: isGeo ? 380 : 300,
        autosize: { type: "fit", contains: "padding", resize: true },
      },
      spec
    );

    var opts = { actions: false, renderer: "canvas", tooltip: true };
    try {
      global.vegaEmbed(plotDiv, fullSpec, opts).catch(function (err) {
        plotDiv.innerHTML = '<div class="solar-chart-fallback">Vega-Lite render failed: ' +
          escapeHtml(err && err.message ? err.message : "unknown error") + "</div>";
      });
    } catch (err) {
      plotDiv.innerHTML = '<div class="solar-chart-fallback">Vega-Lite render failed: ' +
        escapeHtml(err && err.message ? err.message : "unknown error") + "</div>";
    }
  }

  function render(container, payload) {
    if (!container || !payload) return;
    buildHeader(container, payload);

    if (payload.format === "vega-lite") {
      renderVegaLite(container, payload);
      return;
    }
    if (payload.plotly_spec) {
      renderPlotly(container, payload);
      return;
    }
    // Unknown shape — render a graceful notice instead of silently doing nothing
    var notice = document.createElement("div");
    notice.className = "solar-chart-fallback";
    notice.textContent = "Chart payload missing both plotly_spec and Vega-Lite spec.";
    container.appendChild(notice);
  }

  global.ChartRenderer = { render: render };
})(window);
