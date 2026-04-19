/**
 * tool_picker.js — Inline tool toggle pills for Solar AI Chat composer.
 *
 * Two toggle-pills mirror the UX in ChatGPT/Claude:
 *   🌐 Web search   — bias the agent toward external web lookup
 *   📊 Visualize    — bias the agent toward producing chartable data
 *
 * Both are independent (multi-select). With none selected the agent runs in
 * full Auto mode and can still pick any tool.
 *
 * Selection shape sent to backend:
 *   { tool_mode: "auto", allowed_tools: null, tool_hints: ["web_search"?, "visualize"?] }
 *
 * Exposes window.SolarToolPicker:
 *   - mount(container): attach the pill row above the composer (idempotent)
 *   - getSelection():   current selection object
 *   - setHints(hints):  programmatic update
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "solarChatToolHints_v2";

  var TOOLS = [
    {
      id: "web_search",
      label: "Web search",
      tooltip: "Force the agent to pull in up-to-date info from a web search",
      icon:
        '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">' +
        '<path fill="currentColor" d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm6.93 6h-2.95a15.7 15.7 0 0 0-1.38-3.56A8.03 8.03 0 0 1 18.93 8zM12 4.04c.83 1.2 1.48 2.53 1.91 3.96h-3.82c.43-1.43 1.08-2.76 1.91-3.96zM4.26 14a7.85 7.85 0 0 1 0-4h3.38a16.6 16.6 0 0 0 0 4H4.26zm.81 2h2.95c.3 1.28.77 2.5 1.38 3.56A8.03 8.03 0 0 1 5.07 16zm2.95-8H5.07a8.03 8.03 0 0 1 4.33-3.56A15.7 15.7 0 0 0 8.02 8zM12 19.96a13 13 0 0 1-1.91-3.96h3.82A13 13 0 0 1 12 19.96zM14.34 14H9.66a14.6 14.6 0 0 1 0-4h4.68a14.6 14.6 0 0 1 0 4zm.26 5.56c.61-1.06 1.09-2.28 1.38-3.56h2.95a8.03 8.03 0 0 1-4.33 3.56zM16.36 14a16.6 16.6 0 0 0 0-4h3.38a7.85 7.85 0 0 1 0 4h-3.38z"/></svg>',
    },
    {
      id: "visualize",
      label: "Visualize",
      tooltip: "Nudge the agent to fetch chartable time-series / tabular data",
      icon:
        '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">' +
        '<path fill="currentColor" d="M3 3h2v18H3V3zm4 12h2v6H7v-6zm4-6h2v12h-2V9zm4 3h2v9h-2v-9zm4-6h2v15h-2V6z"/></svg>',
    },
  ];

  var hints = loadSavedHints();

  function loadSavedHints() {
    try {
      var raw = global.localStorage && global.localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter(function (h) {
        return h === "web_search" || h === "visualize";
      }) : [];
    } catch (_) {
      return [];
    }
  }

  function saveHints() {
    try {
      global.localStorage &&
        global.localStorage.setItem(STORAGE_KEY, JSON.stringify(hints));
    } catch (_) {}
  }

  function getSelection() {
    return {
      tool_mode: "auto",
      allowed_tools: null,
      tool_hints: hints.slice(),
    };
  }

  function setHints(next) {
    hints = Array.isArray(next) ? next.slice() : [];
    saveHints();
    refreshAll();
  }

  function toggleHint(id) {
    var idx = hints.indexOf(id);
    if (idx >= 0) hints.splice(idx, 1);
    else hints.push(id);
    saveHints();
    refreshAll();
  }

  var roots = [];

  function escapeHtml(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderRoot(root) {
    var html = TOOLS.map(function (t) {
      var active = hints.indexOf(t.id) >= 0 ? " is-active" : "";
      return (
        '<button type="button" class="stp-pill' + active + '" ' +
        'data-hint-id="' + escapeHtml(t.id) + '" ' +
        'title="' + escapeHtml(t.tooltip) + '" ' +
        'aria-pressed="' + (active ? "true" : "false") + '">' +
        '<span class="stp-pill-icon">' + t.icon + "</span>" +
        '<span class="stp-pill-label">' + escapeHtml(t.label) + "</span>" +
        "</button>"
      );
    }).join("");
    html +=
      '<span class="stp-hint-text">' +
      (hints.length
        ? "Agent will use selected tools"
        : "Auto — agent picks tools as needed") +
      "</span>";
    root.innerHTML = html;
  }

  function refreshAll() {
    roots.forEach(renderRoot);
  }

  function mount(container) {
    if (!container) return;
    // Avoid double-mount in the same container
    if (container.querySelector(":scope > .stp-toolbar")) return;

    var root = document.createElement("div");
    root.className = "stp-toolbar";
    // Insert BEFORE the input-wrap (if container is input-wrap itself,
    // sit at the top by prepending; if container is input-wrap's parent,
    // insert just before it).
    if (container.classList && container.classList.contains("solar-chat-input-wrap")) {
      // Put pill row just above the input
      container.parentNode.insertBefore(root, container);
    } else {
      container.appendChild(root);
    }
    roots.push(root);
    renderRoot(root);

    root.addEventListener("click", function (e) {
      var pill = e.target.closest && e.target.closest("[data-hint-id]");
      if (!pill) return;
      toggleHint(pill.getAttribute("data-hint-id"));
    });
  }

  function autoMount() {
    var container = document.querySelector(".solar-chat-input-wrap");
    if (container) mount(container);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", autoMount);
  } else {
    autoMount();
  }

  global.SolarToolPicker = {
    mount: mount,
    getSelection: getSelection,
    setHints: setHints,
  };
})(window);
