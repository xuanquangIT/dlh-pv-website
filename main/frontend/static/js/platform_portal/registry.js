(function () {
  const colors = [
    { border: "#e07b39", bg: "rgba(224,123,57,.1)" }, // orange
    { border: "#1b6ca8", bg: "rgba(27,108,168,.1)" }, // blue
    { border: "#1a8a5a", bg: "rgba(26,138,90,.1)" }, // green
    { border: "#c0392b", bg: "rgba(192,57,43,.1)" } // red
  ];

  let allModels = [];

  const MODEL_HORIZON = {
    'pv.gold.daily_forecast_d1': 'D+1',
    'pv.gold.daily_forecast_d3': 'D+3',
    'pv.gold.daily_forecast_d5': 'D+5',
    'pv.gold.daily_forecast_d7': 'D+7',
  };

  function horizonLabel(modelName) {
    return MODEL_HORIZON[modelName] || modelName || 'Unknown';
  }

  function parseMetricValue(value) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function formatMetricValue(value, decimals, emptyValue, suffix) {
    const parsed = parseMetricValue(value);
    if (parsed === null) return emptyValue;
    return parsed.toFixed(decimals) + (suffix || "");
  }

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
        return `
          <tr class="registry-row" data-model-idx="${idx}" style="cursor:pointer;">
            <td>${horizonLabel(model.model_name)}</td>
            <td>v${model.version}</td>
            <td>${formatMetricValue(model.rmse, 2, '-', '')}</td>
            <td>${formatMetricValue(model.mae, 2, '-', '')}</td>
            <td>${formatMetricValue(model.r2, 4, '-', '')}</td>
            <td>${formatMetricValue(model.mape, 2, '-', '%')}</td>
            <td>${model.created || '-'}</td>
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

    // Normalize for radar: RMSE (MWh ~100-500), MAE (~50-300), R2 (0-1), MAPE (%)
    const datasets = modelsToCompare.map((m, idx) => {
        const c = colors[idx % colors.length];
        const pRmse = parseMetricValue(m.rmse) || 0;
        const pMae  = parseMetricValue(m.mae)  || 0;
        const pR2   = parseMetricValue(m.r2)   || 0;
        const pMape = parseMetricValue(m.mape) || 0;

        // Scale: 0=worst, 100=best for each metric
        const nRMSE = Math.max(0, Math.min(100, (500 - pRmse) / 5));
        const nMAE  = Math.max(0, Math.min(100, (400 - pMae)  / 4));
        const nR2   = Math.max(0, Math.min(100, pR2 * 100));
        const nMAPE = Math.max(0, Math.min(100, 100 - pMape * 2));

        return {
            label: `${horizonLabel(m.model_name)} v${m.version}`,
            data: [nRMSE, nMAE, nR2, nMAPE],
            borderColor: c.border,
            backgroundColor: c.bg,
            pointBackgroundColor: c.border
        };
    });

    new Chart(canvas, {
      type: "radar",
      data: {
        labels: ["RMSE Score", "MAE Score", "R² (%)", "MAPE Score"],
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: { angleLines: { display: true }, min: 0, max: 100 }
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

    const labels = sorted.map(function (m) { return `${horizonLabel(m.model_name)} v${m.version}`; });
    const rmse = sorted.map(function (m) {
      return parseMetricValue(m.rmse);
    });
    const mae = sorted.map(function (m) {
      return parseMetricValue(m.mae);
    });
    const r2 = sorted.map(function (m) {
      return parseMetricValue(m.r2);
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

    const horizon = horizonLabel(model.model_name);
    title.textContent = `Model Details · ${horizon} v${model.version || '?'}`;

    const rmse = formatMetricValue(model.rmse, 2, 'N/A', ' MWh');
    const mae  = formatMetricValue(model.mae,  2, 'N/A', ' MWh');
    const r2   = formatMetricValue(model.r2,   4, 'N/A', '');
    const mape = formatMetricValue(model.mape, 2, 'N/A', '%');

    body.innerHTML = `
      <div class="detail-section">
        <h3 class="detail-section-title">Model Info</h3>
        <div class="detail-row"><span class="detail-label">Horizon</span><span class="detail-value">${horizon}</span></div>
        <div class="detail-row"><span class="detail-label">Model Name</span><span class="detail-value">${model.model_name || 'Unknown'}</span></div>
        <div class="detail-row"><span class="detail-label">Version</span><span class="detail-value">v${model.version || 'Unknown'}</span></div>
        <div class="detail-row"><span class="detail-label">Created</span><span class="detail-value">${model.created || 'Unknown'}</span></div>
      </div>
      <div class="detail-section">
        <h3 class="detail-section-title">Metrics (ALL facilities avg)</h3>
        <div class="detail-row"><span class="detail-label">R²</span><span class="detail-value">${r2}</span></div>
        <div class="detail-row"><span class="detail-label">RMSE</span><span class="detail-value">${rmse}</span></div>
        <div class="detail-row"><span class="detail-label">MAE</span><span class="detail-value">${mae}</span></div>
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
