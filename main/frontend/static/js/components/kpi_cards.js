/**
 * kpi_cards.js — Render KPI card grid from KpiCardsPayload.
 * Exposes window.KpiCards.render(container, payload).
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

  function formatValue(card) {
    var v = card.value;
    if (typeof v !== "number") return escapeHtml(v);
    if (card.format === "integer") return v.toLocaleString();
    if (card.format === "percent") return v.toFixed(1);
    // plain number
    if (Math.abs(v) >= 100) return v.toFixed(1);
    if (Math.abs(v) >= 10) return v.toFixed(2);
    return v.toFixed(3);
  }

  function render(container, payload) {
    if (!container || !payload || !payload.cards || !payload.cards.length) return;
    var cards = payload.cards.map(function (card) {
      var unit = card.unit ? ' <span class="kpi-unit">' + escapeHtml(card.unit) + "</span>" : "";
      var trendIcon = "";
      if (card.trend === "up") trendIcon = '<span class="kpi-trend kpi-trend-up">▲</span>';
      else if (card.trend === "down") trendIcon = '<span class="kpi-trend kpi-trend-down">▼</span>';
      return (
        '<div class="kpi-card">' +
        '<div class="kpi-label">' + escapeHtml(card.label) + "</div>" +
        '<div class="kpi-value">' + formatValue(card) + unit + trendIcon + "</div>" +
        (card.description ? '<div class="kpi-description">' + escapeHtml(card.description) + "</div>" : "") +
        "</div>"
      );
    }).join("");
    var title = payload.title
      ? '<div class="kpi-title">' + escapeHtml(payload.title) + "</div>"
      : "";
    container.innerHTML = title + '<div class="kpi-grid">' + cards + "</div>";
  }

  global.KpiCards = { render: render };
})(window);
