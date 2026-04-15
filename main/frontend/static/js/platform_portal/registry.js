(function () {
  const colors = [
    { border: "#e07b39", bg: "rgba(224,123,57,.1)" }, // orange
    { border: "#1b6ca8", bg: "rgba(27,108,168,.1)" }, // blue
    { border: "#1a8a5a", bg: "rgba(26,138,90,.1)" }, // green
    { border: "#c0392b", bg: "rgba(192,57,43,.1)" } // red
  ];

  let allModels = [];

  async function loadRegistryModels() {
    try {
      const res = await fetch('/model-registry/models-list');
      if (!res.ok) throw new Error('Network response was not ok');
      const data = await res.json();
      allModels = Array.isArray(data) ? data : [];
      
      const tbody = document.getElementById('registry-table-body');
      if (!tbody) return;

      if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">No models registered</td></tr>';
        const summary = document.getElementById('lineage-summary');
        if (summary) summary.textContent = 'No model data available for lineage.';
        return;
      }

      tbody.innerHTML = data.map(function(model, idx) {
        const badgeClass = model.status === "Production" ? "badge-success" : "badge-info";
        let rmse = model.rmse !== null ? parseFloat(model.rmse) : null;
        let mae = model.mae !== null ? parseFloat(model.mae) : null;
        let r2 = model.r2 !== null ? parseFloat(model.r2) : null;
        let mape = model.mape !== null ? parseFloat(model.mape) : null;
        return `
          <tr class="registry-row" data-model-idx="${idx}" style="cursor:pointer;">
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
      renderLineageChart(data);

    } catch (err) {
      console.error("Failed to load registry models", err);
      const tbody = document.getElementById('registry-table-body');
      if (tbody) tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">Error loading models</td></tr>';
    }
  }

  function renderChart(modelsToCompare) {
    const canvas = document.getElementById("compareChart");
    if (!canvas || typeof Chart === "undefined" || modelsToCompare.length === 0) return;

    const existingChart = Chart.getChart(canvas);
    if (existingChart) existingChart.destroy();

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

  function renderLineageChart(models) {
    const canvas = document.getElementById("lineageChart");
    const summary = document.getElementById("lineage-summary");
    if (!canvas || typeof Chart === "undefined") return;
    if (!Array.isArray(models) || models.length === 0) {
      if (summary) summary.textContent = 'No model data available for lineage.';
      return;
    }

    const existingChart = Chart.getChart(canvas);
    if (existingChart) existingChart.destroy();

    const sorted = [...models].sort(function (a, b) {
      return new Date(a.created).getTime() - new Date(b.created).getTime();
    });

    const labels = sorted.map(function (m) { return m.version || "Unknown"; });
    const rmse = sorted.map(function (m) {
      const v = m.rmse !== null ? parseFloat(m.rmse) : null;
      return Number.isFinite(v) ? v : null;
    });
    const mae = sorted.map(function (m) {
      const v = m.mae !== null ? parseFloat(m.mae) : null;
      return Number.isFinite(v) ? v : null;
    });
    const r2 = sorted.map(function (m) {
      const v = m.r2 !== null ? parseFloat(m.r2) : null;
      return Number.isFinite(v) ? v : null;
    });

    const hasAnyMetric = rmse.some(v => v !== null) || mae.some(v => v !== null) || r2.some(v => v !== null);
    if (!hasAnyMetric) {
      if (summary) summary.textContent = 'Lineage data loaded but metrics are missing/invalid.';
      return;
    }

    new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "RMSE",
            data: rmse,
            borderColor: "#c0392b",
            backgroundColor: "rgba(192,57,43,.1)",
            tension: 0.25,
            yAxisID: "y"
          },
          {
            label: "MAE",
            data: mae,
            borderColor: "#e07b39",
            backgroundColor: "rgba(224,123,57,.1)",
            tension: 0.25,
            yAxisID: "y"
          },
          {
            label: "R2",
            data: r2,
            borderColor: "#1b6ca8",
            backgroundColor: "rgba(27,108,168,.1)",
            tension: 0.25,
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false
        },
        scales: {
          y: {
            type: "linear",
            position: "left"
          },
          y1: {
            type: "linear",
            position: "right",
            grid: { drawOnChartArea: false },
            min: 0,
            max: 1
          }
        }
      }
    });

    if (summary) {
      summary.textContent = `Tracking ${sorted.length} versions from ${labels[0]} to ${labels[labels.length - 1]}.`;
    }
  }

  function showModelDetails(model) {
    const modal = document.getElementById('model-detail-modal');
    const title = document.getElementById('model-detail-title');
    const body = document.getElementById('model-detail-body');
    if (!modal || !title || !body || !model) return;

    title.textContent = `Model Details · ${model.version || 'Unknown'}`;

    const badgeClass = model.status === 'Production' ? 'badge-success' : 'badge-info';
    const rmse = model.rmse !== null ? parseFloat(model.rmse).toFixed(3) : 'N/A';
    const mae = model.mae !== null ? parseFloat(model.mae).toFixed(3) : 'N/A';
    const r2 = model.r2 !== null ? parseFloat(model.r2).toFixed(4) : 'N/A';
    const mape = model.mape !== null ? parseFloat(model.mape).toFixed(2) + '%' : 'N/A';

    body.innerHTML = `
      <div class="detail-section">
        <h3 class="detail-section-title">Model Info</h3>
        <div class="detail-row"><span class="detail-label">Version</span><span class="detail-value">${model.version || 'Unknown'}</span></div>
        <div class="detail-row"><span class="detail-label">Algorithm</span><span class="detail-value">${model.algorithm || 'Unknown'}</span></div>
        <div class="detail-row"><span class="detail-label">Status</span><span class="detail-value"><span class="badge ${badgeClass}">${model.status || 'Unknown'}</span></span></div>
        <div class="detail-row"><span class="detail-label">Created</span><span class="detail-value">${model.created || 'Unknown'}</span></div>
      </div>
      <div class="detail-section">
        <h3 class="detail-section-title">Metrics</h3>
        <div class="detail-row"><span class="detail-label">RMSE</span><span class="detail-value">${rmse}</span></div>
        <div class="detail-row"><span class="detail-label">MAE</span><span class="detail-value">${mae}</span></div>
        <div class="detail-row"><span class="detail-label">R2</span><span class="detail-value">${r2}</span></div>
        <div class="detail-row"><span class="detail-label">MAPE</span><span class="detail-value">${mape}</span></div>
      </div>
    `;

    modal.classList.add('open');
  }

  function closeModelDetails() {
    const modal = document.getElementById('model-detail-modal');
    if (modal) modal.classList.remove('open');
  }

  document.addEventListener('DOMContentLoaded', function() {
    loadRegistryModels();

    document.addEventListener('click', function (event) {
      const row = event.target.closest('.registry-row');
      if (row) {
        const idx = Number(row.dataset.modelIdx);
        if (!Number.isNaN(idx)) {
          showModelDetails(allModels[idx]);
        }
        return;
      }

      if (event.target.closest('#model-detail-close')) {
        closeModelDetails();
        return;
      }

      const modal = document.getElementById('model-detail-modal');
      if (modal && event.target === modal) {
        closeModelDetails();
      }
    });
  });

})();
