(function () {
  const colors = {
    solar: "#f4b942",
    blue: "#1b6ca8",
    green: "#1a8a5a",
    red: "#c0392b"
  };

  let selectedHorizon = 1;
  let trainingChart = null;
  let waterfallChart = null;
  let residualsChart = null;

  function updateAccuracyGrade(r2) {
    const letterEl = document.getElementById('accuracy-grade-letter');
    const descEl = document.getElementById('accuracy-grade-desc');
    if (!letterEl || !descEl || isNaN(r2)) return;

    let grade = 'D', color = '#e15759', bg = '#e1575922', desc = '< 0.60 · needs immediate retrain';
    if (r2 >= 0.85) { grade = 'A'; color = '#7fc97f'; bg = '#7fc97f22'; desc = '≥ 0.85 · excellent performance'; }
    else if (r2 >= 0.75) { grade = 'B'; color = '#f6c544'; bg = '#f6c54422'; desc = '≥ 0.75 · operationally usable'; }
    else if (r2 >= 0.60) { grade = 'C'; color = '#ffa24a'; bg = '#ffa24a22'; desc = '≥ 0.60 · underperforming'; }

    letterEl.textContent = grade;
    letterEl.style.color = color;
    letterEl.style.borderColor = color + '77';
    letterEl.style.background = bg;
    descEl.textContent = `R² ${desc}`;
  }

  async function loadMonitoring() {
    try {
      const res = await fetch(`/ml-training/monitoring?horizon=${selectedHorizon}`);
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();

      const pTable = document.getElementById('training-params-table');
      
      if (!data || data.length === 0) {
        if (pTable) pTable.innerHTML = '<tr><td colspan="2" style="text-align:center;">No monitoring data for this horizon</td></tr>';
        return;
      }

      const latest = data[data.length - 1];

      const lbl = document.getElementById('active-model-run');
      if (lbl) lbl.textContent = `v${latest.model_version} · D+${selectedHorizon}`;

      if (pTable) {
        const fmt = (v, d, s = '') => isNaN(parseFloat(v)) ? 'N/A' : parseFloat(v).toFixed(d) + s;
        const kpis = [
          { name: 'Model',       value: latest.model_name || 'N/A' },
          { name: 'Eval Date',   value: latest.date },
          { name: 'R² Score',    value: fmt(latest.r2, 4) },
          { name: 'RMSE (MWh)',  value: fmt(latest.rmse, 2) },
          { name: 'nRMSE',       value: isNaN(parseFloat(latest.rmse)) ? 'N/A' : (parseFloat(latest.rmse) / 45.0 * 100).toFixed(1) + '%' },
          { name: 'MAE (MWh)',   value: fmt(latest.mae, 2) },
          { name: 'MAPE',        value: fmt(latest.mape, 2, '%') },
          { name: 'Skill Score', value: fmt(latest.skill_score, 3) },
        ];
        pTable.innerHTML = kpis.map(k => `<tr><th>${k.name}</th><td>${k.value}</td></tr>`).join('');
      }

      const r2 = parseFloat(latest.r2);
      const bar = document.getElementById('training-progress-bar');
      const txt = document.getElementById('training-progress-text');
      if (bar && txt && !isNaN(r2)) {
        const pct = Math.max(0, Math.min(100, r2 * 100)).toFixed(1);
        bar.style.width = pct + "%";
        txt.textContent = `${pct}% Reliability (R²: ${r2.toFixed(4)})`;
      }

      updateAccuracyGrade(r2);

      renderChart(data);
      await loadResiduals();

    } catch (err) {
      console.error("Failed to load monitoring data", err);
      const pTable = document.getElementById('training-params-table');
      if (pTable) pTable.innerHTML = '<tr><td colspan="2" style="text-align:center;">Error loading data</td></tr>';
    }
  }

  async function loadResiduals() {
    try {
      const residualsRes = await fetch(`/ml-training/residuals?horizon=${selectedHorizon}`);
      const residualsData = await residualsRes.json();
      renderResidualsChart(residualsData);
    } catch (err) {
      console.error("Failed to load residuals", err);
    }
  }

  function renderChart(data) {
    const canvas = document.getElementById("trainingChart");
    if (!canvas || typeof Chart === "undefined") return;

    if (trainingChart) { trainingChart.destroy(); trainingChart = null; }

    const labels = data.map(d => d.date);
    const rmse   = data.map(d => parseFloat(d.rmse));
    const skill  = data.map(d => parseFloat(d.skill_score));
    const r2     = data.map(d => parseFloat(d.r2));

    trainingChart = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "RMSE (MWh)",  data: rmse,  borderColor: colors.red,   tension: 0.3, yAxisID: 'y'  },
          { label: "Skill Score", data: skill, borderColor: colors.blue,  tension: 0.3, yAxisID: 'y1' },
          { label: "R²",          data: r2,    borderColor: colors.green, tension: 0.3, yAxisID: 'y1', borderDash: [4,2] }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          y:  { type: 'linear', position: 'left',  title: { display: true, text: 'RMSE (MWh)' } },
          y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false },
                title: { display: true, text: 'Score' }, min: 0, max: 1 }
        }
      }
    });
  }


  function renderResidualsChart(data) {
    const canvas = document.getElementById("residualsChart");
    if (!canvas || typeof Chart === "undefined") return;

    if (residualsChart) { residualsChart.destroy(); residualsChart = null; }

    const labels = data.map(d => d.date_md);
    const residuals = data.map(d => parseFloat(d.residual));

    residualsChart = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: "Residual (MWh)",
          data: residuals,
          borderColor: "#e59aa8",
          backgroundColor: "rgba(229,154,168,0.2)",
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { title: { display: true, text: 'MWh' } }
        }
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    loadMonitoring();

    document.getElementById('horizon-select')?.addEventListener('change', function () {
      selectedHorizon = parseInt(this.value, 10);
      loadMonitoring();
    });
  });

})();
