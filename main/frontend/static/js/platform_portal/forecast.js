(function () {
  const colors = {
    solar: "#f4b942",
    blue: "#1b6ca8",
    green: "#1a8a5a",
    red: "#c0392b"
  };

  async function loadForecastSummary() {
    try {
      const res = await fetch('/forecast/summary-kpi');
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();
      
      const vEl = document.getElementById('kpi-version');
      if (vEl) vEl.textContent = data.model_version || 'Unknown';

      const container = document.getElementById('forecast-kpis');
      if (!container) return;

      const formatKpi = (val, dec, suffix = '') => 
        (val !== 'N/A' && val !== null && val !== undefined) ? (parseFloat(val).toFixed(dec) + suffix) : 'N/A';

      const kpis = [
        { name: 'MAPE', value: formatKpi(data.mape, 2, '%') },
        { name: 'MAE (MWh)', value: formatKpi(data.mae, 3) },
        { name: 'RMSE (MWh)', value: formatKpi(data.rmse, 3) },
        { name: 'R2 Score', value: formatKpi(data.r2, 4) },
        { name: 'Skill Score', value: formatKpi(data.skill_score, 3) },
        { name: 'Eval Date', value: data.date || 'N/A' }
      ];

      container.innerHTML = kpis.map(function(kpi) {
        return `
          <article class="card" style="padding:12px;">
              <div class="kpi-label">${kpi.name}</div>
              <div class="kpi-value">${kpi.value}</div>
          </article>
        `;
      }).join('');

    } catch (err) {
      console.error("Failed to load forecast summary", err);
      const container = document.getElementById('forecast-kpis');
      if (container) container.innerHTML = '<div>Error loading KPIs</div>';
    }
  }

  async function loadForecastDaily() {
    try {
      const res = await fetch('/forecast/daily');
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();
      
      const tbody = document.getElementById('forecast-table-body');
      if (!tbody) return;

      if (data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No forecast data available</td></tr>';
        return;
      }

      tbody.innerHTML = data.map(function(row) {
        let errStr = "N/A";
        let conf = "High";
        
        let actual = row.actual !== null ? parseFloat(row.actual) : null;
        let predicted = row.predicted !== null ? parseFloat(row.predicted) : null;
        let lower = row.lower !== null ? parseFloat(row.lower) : null;
        let upper = row.upper !== null ? parseFloat(row.upper) : null;

        if (actual !== null && predicted !== null) {
            let err = Math.abs(actual - predicted) / ((actual + 0.001));
            errStr = (err * 100).toFixed(1) + "%";
            if (err > 0.15) conf = "Low";
            else if (err > 0.08) conf = "Medium";
        }
        
        const badgeClass = conf === "High" ? "badge-success" : (conf === "Medium" ? "badge-info" : "badge-warn");

        return `
          <tr>
            <td>${row.date}</td>
            <td>${actual !== null ? actual.toFixed(2) : '-'}</td>
            <td>${predicted !== null ? predicted.toFixed(2) : '-'}</td>
            <td>${lower !== null ? lower.toFixed(2) : '-'}</td>
            <td>${upper !== null ? upper.toFixed(2) : '-'}</td>
            <td>${errStr}</td>
            <td><span class="badge ${badgeClass}">${conf}</span></td>
          </tr>
        `;
      }).join('');

      renderChart(data);

    } catch (err) {
      console.error("Failed to load daily forecast", err);
      const tbody = document.getElementById('forecast-table-body');
      if (tbody) tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">Error loading forecast data</td></tr>';
    }
  }

  function renderChart(data) {
    const canvas = document.getElementById("forecastChart");
    if (!canvas || typeof Chart === "undefined") return;

    const labels = data.map(d => d.date);
    const actual = data.map(d => d.actual !== null ? parseFloat(d.actual) : null);
    const predicted = data.map(d => d.predicted !== null ? parseFloat(d.predicted) : null);

    new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { 
            label: "Actual (MWh)", 
            data: actual, 
            borderColor: colors.solar, 
            backgroundColor: "rgba(244, 185, 66, 0.1)",
            tension: 0.3,
            fill: true
          },
          { 
            label: "Predicted (MWh)", 
            data: predicted, 
            borderColor: colors.blue, 
            borderDash: [5, 3],
            tension: 0.3, 
            fill: false 
          }
        ]
      },
      options: { 
          responsive: true, 
          maintainAspectRatio: false,
          interaction: {
            mode: 'index',
            intersect: false,
          }
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function() {
    loadForecastSummary();
    loadForecastDaily();
  });

})();
