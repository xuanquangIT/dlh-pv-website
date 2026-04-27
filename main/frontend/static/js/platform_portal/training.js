(function () {
  const colors = {
    solar: "#f4b942",
    blue: "#1b6ca8",
    green: "#1a8a5a",
    red: "#c0392b"
  };

  let selectedHorizon = 1;
  let trainingChart = null;

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
          { name: 'RMSE (MWh)', value: fmt(latest.rmse, 2) },
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

      renderChart(data);

    } catch (err) {
      console.error("Failed to load monitoring data", err);
      const pTable = document.getElementById('training-params-table');
      if (pTable) pTable.innerHTML = '<tr><td colspan="2" style="text-align:center;">Error loading data</td></tr>';
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

  document.addEventListener('DOMContentLoaded', function () {
    loadMonitoring();

    document.getElementById('horizon-select')?.addEventListener('change', function () {
      selectedHorizon = parseInt(this.value, 10);
      loadMonitoring();
    });
  });

})();
