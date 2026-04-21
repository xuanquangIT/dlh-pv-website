/**
 * chart_renderer.js — Render Plotly chart from ChartPayload.
 *
 * Exposes window.ChartRenderer.render(container, payload).
 * Requires Plotly.js to be loaded globally (window.Plotly). Gracefully
 * degrades to a no-op with a notice if Plotly is missing.
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

  function render(container, payload) {
    if (!container || !payload || !payload.plotly_spec) return;
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
    // Our wrapper already renders the title, so drop Plotly's duplicate.
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
      // Re-flow when the chat layout reflows (e.g. sidebar collapse, window resize)
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

  global.ChartRenderer = { render: render };
})(window);
