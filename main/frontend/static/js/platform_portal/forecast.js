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
  let lastDailyData = [];

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

  function toWeekValue(dateStr) {
    const d = new Date(dateStr + "T00:00:00");
    const utcDate = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
    const day = utcDate.getUTCDay() || 7;
    utcDate.setUTCDate(utcDate.getUTCDate() + 4 - day);
    const yearStart = new Date(Date.UTC(utcDate.getUTCFullYear(), 0, 1));
    const weekNo = Math.ceil((((utcDate - yearStart) / 86400000) + 1) / 7);
    return `${utcDate.getUTCFullYear()}-W${String(weekNo).padStart(2, "0")}`;
  }

  function weekValueToDate(weekValue) {
    if (!weekValue || !weekValue.includes("-W")) return null;
    const parts = weekValue.split("-W");
    const year = parseInt(parts[0], 10);
    const week = parseInt(parts[1], 10);
    if (!Number.isFinite(year) || !Number.isFinite(week)) return null;

    const jan4 = new Date(Date.UTC(year, 0, 4));
    const jan4Day = jan4.getUTCDay() || 7;
    const monday = new Date(jan4);
    monday.setUTCDate(jan4.getUTCDate() - (jan4Day - 1) + (week - 1) * 7);
    const local = new Date(monday.getUTCFullYear(), monday.getUTCMonth(), monday.getUTCDate());
    return toISODate(local);
  }

  let weekStart = getWeekStart(toISODate(new Date()));

  function updateWeekLabel() {
    const end = getWeekEnd(weekStart);
    const el = document.getElementById("week-range-display");
    if (el) el.textContent = `${formatDisplay(weekStart)} – ${formatDisplay(end)}`;
  }

  function updateWeekPicker() {
    const picker = document.getElementById("week-picker");
    if (picker) picker.value = toWeekValue(weekStart);
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
      if (vEl) {
        const modelName = data.model_name || 'Unknown model';
        const label = data.model_label ? String(data.model_label).trim() : '';
        const version = (data.model_version !== undefined && data.model_version !== null)
          ? `v${data.model_version}`
          : '';
        const alias = data.model_alias ? `alias ${data.model_alias}` : '';
        const labelVersion = `${label} ${version}`.trim();
        const parts = [modelName];
        if (labelVersion) parts.push(labelVersion);
        if (alias) parts.push(alias);
        vEl.textContent = parts.join(' · ');
      }

      const container = document.getElementById('forecast-kpis');
      if (!container) return;

      const formatKpi = (val, dec, suffix = '') =>
        (val !== 'N/A' && val !== null && val !== undefined) ? (parseFloat(val).toFixed(dec) + suffix) : 'N/A';

      const kpis = [
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
      lastDailyData = Array.isArray(data) ? data : [];

      if (!tbody) return;

      if (lastDailyData.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:16px;">No data available for this week</td></tr>`;
        return;
      }

      const ciHalf    = HORIZON_CI[selectedHorizon]          || 0.10;
      const threshLow = HORIZON_CONF_LOW[selectedHorizon]    || 0.15;
      const threshMed = HORIZON_CONF_MEDIUM[selectedHorizon] || 0.08;

      tbody.innerHTML = lastDailyData.map(row => {
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

      renderChart(lastDailyData);

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

    const labels = data.map(d => d.date);
    let actual = data.map(d => d.actual    !== null ? parseFloat(d.actual)    : null);
    let predicted = data.map(d => d.predicted !== null ? parseFloat(d.predicted) : null);

    const allVals = [...actual, ...predicted].filter(v => v !== null && Number.isFinite(v));
    let yOpts = { title: { display: true, text: "MWh" }, beginAtZero: false };
    if (allVals.length > 0) {
      const dMin = Math.min(...allVals);
      const dMax = Math.max(...allVals);
      const range = dMax - dMin || Math.abs(dMax) || 1;
      const pad = range * 0.08;
      yOpts.min = Math.floor(dMin - pad);
      yOpts.max = Math.ceil(dMax + pad);
    }

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
        interaction: { mode: 'index', intersect: false },
        scales: {
          y: yOpts
        }
      }
    });
  }

  // ─── Heatmap & Facility Drill ────────────────────────────────────
  let fdEnergyChart = null;
  let fdWeatherChart = null;
  let fdCfChart = null;

  async function openFacilityDrill(facId) {
      const modal = document.getElementById('facility-drill-modal');
      modal.classList.add('open');
      document.getElementById('facility-drill-loading').style.display = 'block';
      document.getElementById('facility-drill-content').style.display = 'none';
      
      try {
          const res = await fetch(`/forecast/facility-drill/${facId}?horizon=${selectedHorizon}`);
          const data = await res.json();
          
          document.getElementById('fd-code').textContent = data.facility.id;
          document.getElementById('fd-name').textContent = data.facility.name;
          document.getElementById('fd-meta').textContent = `${data.facility.region} · ${data.facility.state} · ${data.facility.capacity} MW · ${data.facility.lat}, ${data.facility.lon}`;
          
          document.getElementById('fd-today-energy').textContent = data.last.energy_mwh_daily.toFixed(1) + " MWh";
          document.getElementById('fd-cf').textContent = data.last.capacity_factor_pct.toFixed(1) + "%";
          document.getElementById('fd-yield').textContent = data.last.specific_yield.toFixed(2) + " kWh/kWp";
          document.getElementById('fd-aqi').textContent = data.last.aqi_value + " (" + data.last.aqi_category + ")";
          
          document.getElementById('fd-grades').innerHTML = data.weeks.map(w => {
            let bg = 'rgba(225,87,89,0.28)';
            if (w.grade === 'A') bg = 'rgba(127,201,127,0.18)';
            else if (w.grade === 'B') bg = 'rgba(246,197,68,0.22)';
            else if (w.grade === 'C') bg = 'rgba(255,150,66,0.25)';
            return `<div style="background:${bg}; padding: 4px 8px; border-radius: 4px; text-align:center;">
                <div style="font-size:10px; color:#556075;">${w.week}</div>
                <div style="font-weight:bold;">${w.r2.toFixed(2)} ${w.grade}</div>
            </div>`;
          }).join('');

          renderFdCharts(data.daily_rows);

          document.getElementById('facility-drill-loading').style.display = 'none';
          document.getElementById('facility-drill-content').style.display = 'block';
      } catch(err) {
          console.error(err);
          document.getElementById('facility-drill-loading').textContent = "Error loading facility data.";
      }
  }

  function renderFdCharts(rows) {
      if (fdEnergyChart) fdEnergyChart.destroy();
      if (fdWeatherChart) fdWeatherChart.destroy();
      if (fdCfChart) fdCfChart.destroy();

      const labels = rows.map(r => r.date_md);

      fdEnergyChart = new Chart(document.getElementById('fdEnergyChart'), {
          type: 'line',
          data: {
              labels,
              datasets: [
                  { label: "Actual", data: rows.map(r=>r.energy_mwh_daily), borderColor: "#f6c544", backgroundColor: "rgba(246,197,68,0.2)", fill:true, tension:0.3 },
                  { label: "Forecast", data: rows.map(r=>r.forecast_mwh), borderColor: "#6aa8ef", borderDash: [5,5], tension:0.3 }
              ]
          },
          options: { responsive: true, maintainAspectRatio: false }
      });

      fdWeatherChart = new Chart(document.getElementById('fdWeatherChart'), {
          type: 'line',
          data: {
              labels,
              datasets: [
                  { label: "Cloud %", data: rows.map(r=>r.cloud_pct), borderColor: "#7da3c4", backgroundColor: "rgba(125,163,196,0.2)", fill:true, tension:0.3, yAxisID: 'y' },
                  { label: "AQI", data: rows.map(r=>r.aqi_value), borderColor: "#e59aa8", tension:0.3, yAxisID: 'y1' }
              ]
          },
          options: { responsive: true, maintainAspectRatio: false, scales: { y: {position:'left'}, y1: {position:'right', grid:{drawOnChartArea:false}} } }
      });

      fdCfChart = new Chart(document.getElementById('fdCfChart'), {
          type: 'line',
          data: {
              labels,
              datasets: [
                  { label: "CF %", data: rows.map(r=>r.capacity_factor_pct), borderColor: "#4fb8a8", backgroundColor: "rgba(79,184,168,0.2)", fill:true, tension:0.3 }
              ]
          },
          options: { responsive: true, maintainAspectRatio: false }
      });
  }

  // ─── Navigation ────────────────────────────────────────────────
  function shiftWeek(direction) {
    weekStart = addDays(weekStart, direction * 7);
    updateWeekLabel();
    updateWeekPicker();
    loadForecastDaily();
  }

  function resetToThisWeek() {
    weekStart = getWeekStart(toISODate(new Date()));
    updateWeekLabel();
    updateWeekPicker();
    loadForecastDaily();
  }

  // ─── Bootstrap ────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', function () {
    updateWeekLabel();
    updateWeekPicker();
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

    document.getElementById('week-picker')?.addEventListener('change', function () {
      const newStart = weekValueToDate(String(this.value || ""));
      if (!newStart) return;
      weekStart = newStart;
      updateWeekLabel();
      loadForecastDaily();
    });

    document.getElementById('facility-drill-close')?.addEventListener('click', () => {
        document.getElementById('facility-drill-modal').classList.remove('open');
    });

    loadForecastSummary();
    loadForecastDaily();
  });

})();
