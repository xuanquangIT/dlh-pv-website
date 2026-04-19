/**
 * tool_status_card.js
 *
 * Feature-flag-aware renderer for inline tool-status cards used by the
 * Solar AI Chat live Task Tracker and the post-response thinking-trace
 * accordion.
 *
 * Exposes `window.ToolStatusCard` with:
 *   - getFeatures(): current UiFeature flag map, seeded from
 *     `window.SOLAR_CHAT_UI_FEATURES` and updated by `updateFeatures(next)`.
 *   - updateFeatures(map): merge new flags (e.g. from a DoneEvent payload).
 *   - canShowTrace(): whether the thinking-trace accordion may render.
 *   - renderRow(task): HTML string for a single task row, honoring flags
 *     (tool-name / duration / argument / error-detail visibility).
 *
 * UiFeature keys (must match backend `schemas/solar_ai_chat/ui_features.py`):
 *   show_tool_names, show_tool_arguments, show_tool_duration_ms,
 *   show_thinking_trace, show_memory_results, show_tool_error_detail.
 */
(function (global) {
  "use strict";

  var DEFAULT_FEATURES = {
    show_tool_names: false,
    show_tool_arguments: false,
    show_tool_duration_ms: false,
    show_thinking_trace: false,
    show_memory_results: false,
    show_tool_error_detail: false
  };

  var features = Object.assign({}, DEFAULT_FEATURES, global.SOLAR_CHAT_UI_FEATURES || {});

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function getFeatures() {
    return Object.assign({}, features);
  }

  function updateFeatures(next) {
    if (!next || typeof next !== "object") return;
    Object.keys(next).forEach(function (key) {
      if (Object.prototype.hasOwnProperty.call(DEFAULT_FEATURES, key)) {
        features[key] = Boolean(next[key]);
      }
    });
  }

  function canShowTrace() {
    return Boolean(features.show_thinking_trace);
  }

  function canShowToolNames() {
    return Boolean(features.show_tool_names);
  }

  function canShowDuration() {
    return Boolean(features.show_tool_duration_ms);
  }

  function canShowArguments() {
    return Boolean(features.show_tool_arguments);
  }

  function canShowErrorDetail() {
    return Boolean(features.show_tool_error_detail);
  }

  function iconFor(status) {
    if (status === "running") {
      return '<span class="task-tracker-spinner" aria-hidden="true"></span>';
    }
    if (status === "ok" || status === "success") {
      return '<span class="task-tracker-icon ok" aria-hidden="true">\u2713</span>';
    }
    if (status === "error" || status === "warning") {
      return '<span class="task-tracker-icon error" aria-hidden="true">\u2715</span>';
    }
    if (status === "denied") {
      return '<span class="task-tracker-icon denied" aria-hidden="true">\u2298</span>';
    }
    return '<span class="task-tracker-icon skipped" aria-hidden="true">\u2013</span>';
  }

  /**
   * Render a single task/step row.
   * @param {object} task - { step, tool_name, label, status, duration_ms, detail, args }
   * @returns {string} HTML — empty when the role cannot see tool names at all.
   */
  function renderRow(task) {
    if (!task) return "";
    if (!canShowToolNames()) {
      // No tool visibility at all → render a generic placeholder so the user
      // still sees activity, but without tool identity.
      if (task.status === "running") {
        return '<div class="task-tracker-row running" aria-live="polite">' +
          iconFor("running") +
          '<span class="task-tracker-label">Working\u2026</span>' +
          '</div>';
      }
      return "";
    }

    var rowClass = "task-tracker-row" + (task.status === "running" ? " running" : "");
    var label = task.label || task.tool_name || task.step || "";
    var parts = [
      '<div class="' + rowClass + '" aria-live="polite">',
      "  " + iconFor(task.status),
      '  <span class="task-tracker-label">' + escapeHtml(label) + "</span>"
    ];

    if (canShowDuration() && task.duration_ms != null && task.duration_ms !== "") {
      parts.push(' <span class="task-tracker-dur">' + escapeHtml(task.duration_ms) + "ms</span>");
    } else if (task.detail) {
      // `detail` in a thinking_trace is a short summary (e.g. "7 metrics retrieved")
      // — not PII — so we let it through even when duration is hidden.
      parts.push(' <span class="task-tracker-dur">' + escapeHtml(task.detail) + "</span>");
    }

    if (canShowArguments() && task.args) {
      var argsText = typeof task.args === "string" ? task.args : JSON.stringify(task.args);
      parts.push(' <code class="task-tracker-args">' + escapeHtml(argsText) + "</code>");
    }

    if (task.status === "error" && canShowErrorDetail() && task.error_detail) {
      parts.push(' <span class="task-tracker-error">' + escapeHtml(task.error_detail) + "</span>");
    }

    parts.push("</div>");
    return parts.join("");
  }

  global.ToolStatusCard = {
    getFeatures: getFeatures,
    updateFeatures: updateFeatures,
    canShowTrace: canShowTrace,
    canShowToolNames: canShowToolNames,
    canShowDuration: canShowDuration,
    canShowArguments: canShowArguments,
    canShowErrorDetail: canShowErrorDetail,
    renderRow: renderRow
  };
})(window);
