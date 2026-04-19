/**
 * data_table.js — Interactive DataTable renderer for Solar AI Chat.
 *
 * Exposes window.DataTable.render(container, payload) where `payload` is the
 * `DataTablePayload` dict emitted by the backend (columns, rows, etc.).
 *
 * Features: sort per column, inline text search/filter, pagination, CSV
 * export. Pure vanilla JS — no deps.
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

  function formatCell(value, column) {
    if (value == null) return "";
    if (column.type === "number" || column.type === "integer") {
      var num = Number(value);
      if (!isNaN(num)) {
        if (column.type === "integer") return num.toLocaleString();
        if (Math.abs(num) >= 100) return num.toFixed(2);
        if (Math.abs(num) >= 1) return num.toFixed(3);
        return num.toFixed(4);
      }
    }
    if (column.type === "date" || column.type === "datetime") {
      return String(value);
    }
    return String(value);
  }

  function compareValues(a, b, type) {
    var av = a == null ? "" : a;
    var bv = b == null ? "" : b;
    if (type === "number" || type === "integer") {
      var na = Number(av);
      var nb = Number(bv);
      if (isNaN(na)) na = -Infinity;
      if (isNaN(nb)) nb = -Infinity;
      return na - nb;
    }
    return String(av).localeCompare(String(bv), undefined, { numeric: true });
  }

  function toCsv(columns, rows) {
    function cell(v) {
      if (v == null) return "";
      var s = String(v);
      if (s.indexOf(",") >= 0 || s.indexOf('"') >= 0 || s.indexOf("\n") >= 0) {
        return '"' + s.replace(/"/g, '""') + '"';
      }
      return s;
    }
    var header = columns.map(function (c) { return cell(c.label); }).join(",");
    var lines = rows.map(function (row) {
      return columns.map(function (c) { return cell(row[c.key]); }).join(",");
    });
    return header + "\n" + lines.join("\n");
  }

  function downloadCsv(filename, content) {
    var blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function render(container, payload) {
    if (!container || !payload || !payload.rows || !payload.columns) return;

    var columns = payload.columns;
    var rows = payload.rows.slice();
    var state = {
      sortKey: null,
      sortDir: "asc",
      filter: "",
      page: 1,
      pageSize: payload.page_size || 25
    };

    container.classList.add("solar-data-table");
    container.innerHTML = "";

    var header = document.createElement("div");
    header.className = "sdt-header";
    header.innerHTML =
      '<div class="sdt-title">' + escapeHtml(payload.title || "Data") + "</div>" +
      (payload.description
        ? '<div class="sdt-desc">' + escapeHtml(payload.description) + "</div>"
        : "");
    container.appendChild(header);

    var controls = document.createElement("div");
    controls.className = "sdt-controls";
    controls.innerHTML =
      (payload.filterable !== false
        ? '<input type="text" class="sdt-filter" placeholder="Search…" />'
        : "") +
      (payload.exportable !== false
        ? '<button type="button" class="sdt-export">Export CSV</button>'
        : "");
    container.appendChild(controls);

    var tableWrap = document.createElement("div");
    tableWrap.className = "sdt-wrap";
    container.appendChild(tableWrap);

    var footer = document.createElement("div");
    footer.className = "sdt-footer";
    container.appendChild(footer);

    function currentRows() {
      var filtered = rows;
      if (state.filter) {
        var q = state.filter.toLowerCase();
        filtered = rows.filter(function (row) {
          return columns.some(function (c) {
            var v = row[c.key];
            return v != null && String(v).toLowerCase().indexOf(q) >= 0;
          });
        });
      }
      if (state.sortKey) {
        var col = columns.find(function (c) { return c.key === state.sortKey; }) || { type: "string" };
        filtered = filtered.slice().sort(function (a, b) {
          var cmp = compareValues(a[col.key], b[col.key], col.type);
          return state.sortDir === "asc" ? cmp : -cmp;
        });
      }
      return filtered;
    }

    function draw() {
      var displayed = currentRows();
      var total = displayed.length;
      var paginated = payload.paginated !== false;
      var pageSize = paginated ? state.pageSize : total;
      var totalPages = Math.max(1, Math.ceil(total / pageSize));
      if (state.page > totalPages) state.page = totalPages;
      if (state.page < 1) state.page = 1;
      var start = (state.page - 1) * pageSize;
      var pageRows = paginated ? displayed.slice(start, start + pageSize) : displayed;

      var thead = "<thead><tr>" + columns.map(function (c) {
        var isSorted = state.sortKey === c.key;
        var arrow = isSorted ? (state.sortDir === "asc" ? " ▲" : " ▼") : "";
        var labelWithUnit = c.unit ? c.label + " (" + c.unit + ")" : c.label;
        return '<th data-key="' + escapeHtml(c.key) + '" class="' +
          (payload.sortable !== false ? "sdt-sortable" : "") +
          '">' + escapeHtml(labelWithUnit) + arrow + "</th>";
      }).join("") + "</tr></thead>";

      var tbody = "<tbody>" + pageRows.map(function (row) {
        return "<tr>" + columns.map(function (c) {
          return "<td>" + escapeHtml(formatCell(row[c.key], c)) + "</td>";
        }).join("") + "</tr>";
      }).join("") + "</tbody>";

      if (pageRows.length === 0) {
        tbody = '<tbody><tr><td class="sdt-empty" colspan="' + columns.length + '">No rows.</td></tr></tbody>';
      }

      tableWrap.innerHTML = "<table class=\"sdt-table\">" + thead + tbody + "</table>";

      footer.innerHTML =
        '<div class="sdt-info">' +
        escapeHtml(String(total)) + " row" + (total === 1 ? "" : "s") +
        (total !== rows.length ? " (filtered from " + rows.length + ")" : "") +
        "</div>" +
        (paginated && totalPages > 1
          ? '<div class="sdt-pager">' +
              '<button type="button" class="sdt-prev"' + (state.page <= 1 ? " disabled" : "") + ">Prev</button>" +
              '<span class="sdt-page">Page ' + state.page + " / " + totalPages + "</span>" +
              '<button type="button" class="sdt-next"' + (state.page >= totalPages ? " disabled" : "") + ">Next</button>" +
            "</div>"
          : "");
    }

    if (payload.sortable !== false) {
      tableWrap.addEventListener("click", function (e) {
        var th = e.target.closest && e.target.closest("th.sdt-sortable");
        if (!th) return;
        var key = th.getAttribute("data-key");
        if (!key) return;
        if (state.sortKey === key) {
          state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = key;
          state.sortDir = "asc";
        }
        draw();
      });
    }

    var filterInput = controls.querySelector(".sdt-filter");
    if (filterInput) {
      filterInput.addEventListener("input", function (e) {
        state.filter = e.target.value || "";
        state.page = 1;
        draw();
      });
    }

    var exportBtn = controls.querySelector(".sdt-export");
    if (exportBtn) {
      exportBtn.addEventListener("click", function () {
        downloadCsv(
          (payload.title || "solar-chat-data").replace(/\s+/g, "_") + ".csv",
          toCsv(columns, currentRows())
        );
      });
    }

    footer.addEventListener("click", function (e) {
      if (e.target.classList.contains("sdt-prev")) {
        state.page -= 1;
        draw();
      } else if (e.target.classList.contains("sdt-next")) {
        state.page += 1;
        draw();
      }
    });

    draw();
  }

  global.DataTable = { render: render };
})(window);
