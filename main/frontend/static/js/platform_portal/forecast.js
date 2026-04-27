(function () {
  const colors = {
    solar: "#f4b942",
    blue: "#1b6ca8",
    green: "#1a8a5a",
    red: "#c0392b"
  };

  // CI half-width and confidence thresholds vary by horizon
  const HORIZON_CI          = { 1: 0.07, 3: 0.10, 5: 0.13, 7: 0.17 };
  const HORIZON_CONF_LOW    = { 1: 0.10, 3: 0.15, 5: 0.20, 7: 0.25 };
  const HORIZON_CONF_MEDIUM = { 1: 0.05, 3: 0.08, 5: 0.12, 7: 0.15 };

  let selectedHorizon = 1;

  // ─── Week helpers ──────────────────────────────────────────────
  function toISODate(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }

  function addDays(dateStr, n) {
    const d = new Date(dateStr + "T00:00:00");
    d.setDate(d.getDate() + n);
    return toISODate(d);
  }

  function getWeekStart(dateStr) {
    const d = new Date(dateStr + "T00:00:00");
    const day = d.getDay();
    const diff = (day === 0) ? -6 : 1 - day;
    d.setDate(d.getDate() + diff);
    return toISODate(d);
  }

  function getWeekEnd(weekStartStr) {
    return addDays(weekStartStr, 6);
  }

  function formatDisplay(dateStr) {
    const d = new Date(dateStr + "T00:00:00");
    return d.toLocaleDateString("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" });
  }

  let weekStart = getWeekStart(toISODate(new Date()));

  function updateWeekLabel() {
    const end = getWeekEnd(weekStart);
    const el = document.getElementById("week-range-display");
    if (el) el.textContent = `${formatDisplay(weekStart)} – ${formatDisplay(end)}`;
  }

  function updateHorizonLabel() {
    const el = document.getElementById('kpi-horizon');
    if (el) el.textContent = selectedHorizon;
  }

  // ─── Load KPI summary ──────────────────────────────────────────
  async function loadForecastSummary() {
    try {
      const res = await fetch(`/forecast/summary-kpi?horizon=${selectedHorizon}`);
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();

      const vEl = document.getElementById('kpi-version');
      if (vEl) vEl.textContent = data.model_version || 'Unknown';

      const container = document.getElementById('forecast-kpis');
      if (!container) return;

      const formatKpi = (val, dec, suffix = '') =>
        (val !== 'N/A' && val !== null && val !== undefined) ? (parseFloat(val).toFixed(dec) + suffix) : 'N/A';

      const kpis = [
        { name: 'MAPE',        value: formatKpi(data.mape, 2, '%') },
        { name: 'MAE (MWh)',   value: formatKpi(data.mae, 3) },
        { name: 'RMSE (MWh)',  value: formatKpi(data.rmse, 3) },
        { name: 'R2 Score',    value: formatKpi(data.r2, 4) },
        { name: 'Skill Score', value: formatKpi(data.skill_score, 3) },
        { name: 'Eval Date',   value: data.date || 'N/A' }
      ];

      container.innerHTML = kpis.map(kpi => `
        <article class="card" style="padding:12px;">
          <div class="kpi-label">${kpi.name}</div>
          <div class="kpi-value">${kpi.value}</div>
        </article>
      `).join('');

    } catch (err) {
      console.error("Failed to load forecast summary", err);
      const container = document.getElementById('forecast-kpis');
      if (container) container.innerHTML = '<div>Error loading KPIs</div>';
    }
  }

  // ─── Load daily forecast ───────────────────────────────────────
  let forecastChart = null;

  async function loadForecastDaily() {
    const start = weekStart;
    const end   = getWeekEnd(weekStart);

    const tbody = document.getElementById('forecast-table-body');
    if (tbody) tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px;">Loading...</td></tr>`;

    try {
      const params = new URLSearchParams({ start_date: start, end_date: end, horizon: selectedHorizon });
      const res = await fetch(`/forecast/daily?${params}`);
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();

      if (!tbody) return;

      if (data.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:16px;">No data available for this week</td></tr>`;
        return;
      }

      const ciHalf    = HORIZON_CI[selectedHorizon]          || 0.10;
      const threshLow = HORIZON_CONF_LOW[selectedHorizon]    || 0.15;
      const threshMed = HORIZON_CONF_MEDIUM[selectedHorizon] || 0.08;

      tbody.innerHTML = data.map(row => {
        let errStr = "N/A";
        let conf   = "High";

        const actual    = row.actual    !== null ? parseFloat(row.actual)    : null;
        const predicted = row.predicted !== null ? parseFloat(row.predicted) : null;
        const lower = predicted !== null ? predicted * (1 - ciHalf) : null;
        const upper = predicted !== null ? predicted * (1 + ciHalf) : null;

        if (actual !== null && predicted !== null) {
          const err = Math.abs(actual - predicted) / (actual + 0.001);
          errStr = (err * 100).toFixed(1) + "%";
          if (err > threshLow) conf = "Low";
          else if (err > threshMed) conf = "Medium";
        }

        const badgeClass = conf === "High" ? "badge-success" : (conf === "Medium" ? "badge-info" : "badge-warn");

        return `
          <tr>
            <td>${row.date}</td>
            <td>${actual    !== null ? actual.toFixed(2)    : '-'}</td>
            <td>${predicted !== null ? predicted.toFixed(2) : '-'}</td>
            <td>${lower     !== null ? lower.toFixed(2)     : '-'}</td>
            <td>${upper     !== null ? upper.toFixed(2)     : '-'}</td>
            <td>${errStr}</td>
            <td><span class="badge ${badgeClass}">${conf}</span></td>
          </tr>
        `;
      }).join('');

      renderChart(data);

    } catch (err) {
      console.error("Failed to load daily forecast", err);
      if (tbody) tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;">Error loading forecast data</td></tr>`;
    }
  }

  // ─── Render chart ──────────────────────────────────────────────
  function renderChart(data) {
    const canvas = document.getElementById("forecastChart");
    if (!canvas || typeof Chart === "undefined") return;

    if (forecastChart) {
      forecastChart.destroy();
      forecastChart = null;
    }

    const labels    = data.map(d => d.date);
    const actual    = data.map(d => d.actual    !== null ? parseFloat(d.actual)    : null);
    const predicted = data.map(d => d.predicted !== null ? parseFloat(d.predicted) : null);

    forecastChart = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Actual (MWh)",
            data: actual,
            borderColor: colors.solar,
            backgroundColor: "rgba(244, 185, 66, 0.1)",
            tension: 0.3,
            fill: true,
            spanGaps: false
          },
          {
            label: "Predicted (MWh)",
            data: predicted,
            borderColor: colors.blue,
            borderDash: [5, 3],
            tension: 0.3,
            fill: false,
            spanGaps: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false }
      }
    });
  }

  // ─── Navigation ────────────────────────────────────────────────
  function shiftWeek(direction) {
    weekStart = addDays(weekStart, direction * 7);
    updateWeekLabel();
    loadForecastDaily();
  }

  function resetToThisWeek() {
    weekStart = getWeekStart(toISODate(new Date()));
    updateWeekLabel();
    loadForecastDaily();
  }

  // ─── Bootstrap ────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', function () {
    updateWeekLabel();
    updateHorizonLabel();

    document.getElementById('btn-prev-week')  ?.addEventListener('click', () => shiftWeek(-1));
    document.getElementById('btn-next-week')  ?.addEventListener('click', () => shiftWeek(+1));
    document.getElementById('btn-reset-range')?.addEventListener('click', resetToThisWeek);
    document.getElementById('btn-refresh')    ?.addEventListener('click', () => { loadForecastSummary(); loadForecastDaily(); });

    document.getElementById('horizon-select')?.addEventListener('change', function () {
      selectedHorizon = parseInt(this.value, 10);
      updateHorizonLabel();
      loadForecastSummary();
      loadForecastDaily();
    });

    loadForecastSummary();
    loadForecastDaily();
  });

})();
