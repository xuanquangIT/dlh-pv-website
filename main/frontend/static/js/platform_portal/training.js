(function () {
  const colors = {
    solar: "#f4b942",
    blue: "#1b6ca8",
    green: "#1a8a5a",
    red: "#c0392b"
  };

  async function loadMonitoring() {
    try {
      const res = await fetch('/ml-training/monitoring');
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();
      
      if (!data || data.length === 0) {
        document.getElementById('training-params-table').innerHTML = '<tr><td colspan="2" style="text-align:center;">No monitoring data available</td></tr>';
        return;
      }

      const latest = data[data.length - 1]; // Assume ordered asc
      
      const lbl = document.getElementById('active-model-run');
      if (lbl) lbl.textContent = "Version: " + latest.model_version;

      const pTable = document.getElementById('training-params-table');
      if (pTable) {
          const kpis = [
            { name: 'Eval Date', value: latest.date },
            { name: 'RMSE (MWh)', value: parseFloat(latest.rmse).toFixed(3) },
            { name: 'MAE (MWh)', value: parseFloat(latest.mae).toFixed(3) },
            { name: 'MAPE', value: parseFloat(latest.mape).toFixed(2) + '%' },
            { name: 'Skill Score', value: parseFloat(latest.skill_score).toFixed(3) }
          ];
          pTable.innerHTML = kpis.map(k => `<tr><th>${k.name}</th><td>${k.value}</td></tr>`).join('');
      }

      const r2 = parseFloat(latest.r2);
      const bar = document.getElementById('training-progress-bar');
      const txt = document.getElementById('training-progress-text');
      if (bar && txt && !isNaN(r2)) {
          let pct = Math.max(0, Math.min(100, r2 * 100)).toFixed(1);
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

    const labels = data.map(d => d.date);
    const rmse = data.map(d => parseFloat(d.rmse));
    const skill = data.map(d => parseFloat(d.skill_score));

    new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { 
            label: "RMSE (MWh)", 
            data: rmse, 
            borderColor: colors.red, 
            tension: 0.3,
            yAxisID: 'y'
          },
          { 
            label: "Skill Score", 
            data: skill, 
            borderColor: colors.blue, 
            tension: 0.3,
            yAxisID: 'y1'
          }
        ]
      },
      options: { 
          responsive: true, 
          maintainAspectRatio: false,
          interaction: {
            mode: 'index',
            intersect: false,
          },
          scales: {
              y: {
                  type: 'linear',
                  display: true,
                  position: 'left',
              },
              y1: {
                  type: 'linear',
                  display: true,
                  position: 'right',
                  grid: { drawOnChartArea: false }
              }
          }
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function() {
    loadMonitoring();
  });

})();
