(function () {
  const colors = [
    { border: "#e07b39", bg: "rgba(224,123,57,.1)" }, // orange
    { border: "#1b6ca8", bg: "rgba(27,108,168,.1)" }, // blue
    { border: "#1a8a5a", bg: "rgba(26,138,90,.1)" }, // green
    { border: "#c0392b", bg: "rgba(192,57,43,.1)" } // red
  ];

  async function loadRegistryModels() {
    try {
      const res = await fetch('/model-registry/models-list');
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();
      
      const tbody = document.getElementById('registry-table-body');
      if (!tbody) return;

      if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">No models registered</td></tr>';
        return;
      }

      tbody.innerHTML = data.map(function(model) {
        const badgeClass = model.status === "Production" ? "badge-success" : "badge-info";
        let rmse = model.rmse !== null ? parseFloat(model.rmse) : null;
        let mae = model.mae !== null ? parseFloat(model.mae) : null;
        let r2 = model.r2 !== null ? parseFloat(model.r2) : null;
        let mape = model.mape !== null ? parseFloat(model.mape) : null;
        return `
          <tr>
            <td>${model.version}</td>
            <td>${model.algorithm || 'Unknown'}</td>
            <td>${rmse !== null ? rmse.toFixed(3) : '-'}</td>
            <td>${mae !== null ? mae.toFixed(3) : '-'}</td>
            <td>${r2 !== null ? r2.toFixed(4) : '-'}</td>
            <td>${mape !== null ? mape.toFixed(2) + '%' : '-'}</td>
            <td>${model.created}</td>
            <td><span class="badge ${badgeClass}">${model.status}</span></td>
          </tr>
        `;
      }).join('');

      renderChart(data.slice(0, 4)); // Compare top 4 models

    } catch (err) {
      console.error("Failed to load registry models", err);
      const tbody = document.getElementById('registry-table-body');
      if (tbody) tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">Error loading models</td></tr>';
    }
  }

  function renderChart(modelsToCompare) {
    const canvas = document.getElementById("compareChart");
    if (!canvas || typeof Chart === "undefined" || modelsToCompare.length === 0) return;

    const datasets = modelsToCompare.map((m, idx) => {
        const c = colors[idx % colors.length];
        
        // Normalize metrics for Radar (higher the better vs lower the better)
        // Here we just plot raw values, though they have different scales.
        // Usually, in a radar chart, they should be normalized to 0-100.
        // For demonstration logic based on typical scaled PV metrics:
        const pRmse = parseFloat(m.rmse) || 0;
        const pMae = parseFloat(m.mae) || 0;
        const pR2 = parseFloat(m.r2) || 0;
        const pMape = parseFloat(m.mape) || 0;

        const nRMSE = Math.max(0, 100 - (pRmse * 1000)); // lower rmse = higher score
        const nMAE = Math.max(0, 100 - (pMae * 1500));
        const nR2 = pR2 * 100;
        const nMAPE = Math.max(0, 100 - (pMape * 10)); // pseudo scoring
        
        return {
            label: m.version,
            data: [nRMSE, nMAE, nR2, nMAPE, 80, 75], // adding speed and mem pseudo-stats
            borderColor: c.border,
            backgroundColor: c.bg,
            pointBackgroundColor: c.border
        };
    });

    new Chart(canvas, {
      type: "radar",
      data: {
        labels: ["Norm RMSE", "Norm MAE", "R2 (%)", "Norm MAPE", "Inference Speed", "Mem Effic"],
        datasets: datasets
      },
      options: { 
          responsive: true, 
          maintainAspectRatio: false,
          scales: {
              r: {
                  angleLines: { display: false },
                  suggestedMin: 50,
                  suggestedMax: 100
              }
          }
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function() {
    loadRegistryModels();
  });

})();
