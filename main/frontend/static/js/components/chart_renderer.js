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

  function renderLeafletMap(container, payload) {
    var mapDiv = document.createElement("div");
    mapDiv.className = "solar-chart-plot solar-chart-leaflet";
    mapDiv.style.height = "440px";
    mapDiv.style.width = "100%";
    mapDiv.style.borderRadius = "8px";
    mapDiv.style.overflow = "hidden";
    container.appendChild(mapDiv);

    if (!global.L) {
      mapDiv.innerHTML = '<div class="solar-chart-fallback">Map unavailable: Leaflet not loaded.</div>';
      return;
    }

    var points = Array.isArray(payload.points) ? payload.points : [];
    if (!points.length) {
      mapDiv.innerHTML = '<div class="solar-chart-fallback">No geographic points to plot.</div>';
      return;
    }

    try {
      var map = global.L.map(mapDiv, {
        scrollWheelZoom: true,
        zoomControl: true,
      });

      // OpenStreetMap tiles — free, no key required.
      global.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      }).addTo(map);

      // Compute size scaling so the largest point is ~22px and smallest ~6px.
      var sizes = points.map(function (p) {
        return typeof p.size_value === "number" ? p.size_value : null;
      }).filter(function (v) { return v !== null; });
      var minSize = sizes.length ? Math.min.apply(null, sizes) : 0;
      var maxSize = sizes.length ? Math.max.apply(null, sizes) : 1;
      var sizeRange = Math.max(maxSize - minSize, 1);

      function radiusFor(p) {
        if (typeof p.size_value !== "number") return 7;
        var t = (p.size_value - minSize) / sizeRange;
        return 6 + t * 16;
      }

      var bounds = [];
      points.forEach(function (p) {
        if (typeof p.lat !== "number" || typeof p.lng !== "number") return;
        bounds.push([p.lat, p.lng]);
        var marker = global.L.circleMarker([p.lat, p.lng], {
          radius: radiusFor(p),
          color: "#fff",
          weight: 1.5,
          fillColor: "#FFB100",
          fillOpacity: 0.85,
        }).addTo(map);

        // Build popup HTML from label + size + extra attrs.
        var rows = [];
        if (p.label) {
          rows.push('<div style="font-weight:600;margin-bottom:4px;">' +
                    escapeHtml(p.label) + '</div>');
        }
        if (payload.size_field && typeof p.size_value === "number") {
          rows.push('<div><b>' + escapeHtml(payload.size_field) + ':</b> ' +
                    p.size_value.toLocaleString(undefined, { maximumFractionDigits: 2 }) + '</div>');
        }
        var attrs = p.attrs || {};
        Object.keys(attrs).slice(0, 8).forEach(function (k) {
          if (k === payload.label_field || k === payload.size_field) return;
          var v = attrs[k];
          if (v === null || v === undefined || v === "") return;
          rows.push('<div><b>' + escapeHtml(k) + ':</b> ' + escapeHtml(v) + '</div>');
        });
        marker.bindPopup(rows.join(""));
        marker.bindTooltip(p.label || "", { direction: "top", offset: [0, -6] });
      });

      if (bounds.length) {
        map.fitBounds(bounds, { padding: [40, 40] });
      } else {
        map.setView([0, 0], 2);
      }

      // Resize-aware: when chart container becomes visible, refresh tiles.
      setTimeout(function () { try { map.invalidateSize(); } catch (_) {} }, 50);
    } catch (err) {
      mapDiv.innerHTML = '<div class="solar-chart-fallback">Map render failed: ' +
        escapeHtml(err && err.message ? err.message : "unknown error") + '</div>';
    }
  }

  function render(container, payload) {
    if (!container || !payload) return;
    buildHeader(container, payload);

    if (payload.format === "leaflet-map") {
      renderLeafletMap(container, payload);
      return;
    }
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
