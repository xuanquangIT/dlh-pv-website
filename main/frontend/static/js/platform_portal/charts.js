(function () {
  const colors = {
    solar: "#f4b942",
    green: "#1a8a5a",
    blue: "#1b6ca8",
    orange: "#e07b39",
    red: "#c0392b"
  };

  function initDashboardCharts() {
    const energyCanvas = document.getElementById("energyChart");
    const ratioCanvas = document.getElementById("ratioChart");
    if (!energyCanvas || !ratioCanvas || typeof Chart === "undefined") return;

    new Chart(energyCanvas, {
      type: "line",
      data: {
        labels: ["Jan 2", "Jan 3", "Jan 4", "Jan 5", "Jan 6", "Jan 7", "Jan 8"],
        datasets: [
          {
            label: "Actual (MWh)",
            data: [612, 588, 701, 653, 724, 698, 745],
            borderColor: colors.solar,
            backgroundColor: "rgba(244, 185, 66, 0.1)",
            fill: true,
            tension: 0.35
          },
          {
            label: "Predicted (MWh)",
            data: [590, 610, 690, 660, 710, 720, 750],
            borderColor: colors.blue,
            borderDash: [5, 3],
            fill: false,
            tension: 0.35
          }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });

    new Chart(ratioCanvas, {
      type: "doughnut",
      data: {
        labels: ["Valid", "Warning", "Invalid"],
        datasets: [{ data: [82.1, 12.4, 5.5], backgroundColor: [colors.green, colors.solar, colors.red] }]
      },
      options: { responsive: true, maintainAspectRatio: false, cutout: "70%" }
    });
  }

  function initTrainingChart() {
    const trainingCanvas = document.getElementById("trainingChart");
    if (!trainingCanvas || typeof Chart === "undefined") return;

    const labels = Array.from({ length: 20 }, function (_, i) { return i * 25; });
    const trainLoss = labels.map(function (x) { return +(0.9 * Math.exp(-x / 200) + 0.04).toFixed(4); });
    const valLoss = trainLoss.map(function (x) { return +(x * 1.1).toFixed(4); });

    new Chart(trainingCanvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { label: "Train Loss", data: trainLoss, borderColor: colors.solar, tension: 0.3 },
          { label: "Val Loss", data: valLoss, borderColor: colors.blue, tension: 0.3 }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  function initForecastChart() {
    const canvas = document.getElementById("forecastChart");
    if (!canvas || typeof Chart === "undefined") return;

    const labels = Array.from({ length: 24 }, function (_, i) { return String(i).padStart(2, "0") + ":00"; });
    const predicted = labels.map(function (_, i) {
      return i >= 6 && i <= 18 ? Math.round((Math.sin((i - 6) / 12 * Math.PI) * 220) + 20) : Math.round(Math.random() * 8);
    });

    new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { label: "Predicted", data: predicted, borderColor: colors.blue, tension: 0.3, fill: false }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  function initCompareChart() {
    const canvas = document.getElementById("compareChart");
    if (!canvas || typeof Chart === "undefined") return;
    new Chart(canvas, {
      type: "radar",
      data: {
        labels: ["RMSE", "R2", "Speed", "Memory"],
        datasets: [
          { label: "GBT-v4.2", data: [92, 94, 80, 75], borderColor: colors.orange, backgroundColor: "rgba(224,123,57,.1)" },
          { label: "GBT-v4.1", data: [84, 88, 82, 78], borderColor: colors.blue, backgroundColor: "rgba(27,108,168,.1)" }
        ]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  function bindTopbarActions() {
    const runButton = document.getElementById("run-pipeline-btn");
    if (!runButton) return;
    runButton.addEventListener("click", function () {
      runButton.textContent = "Running";
      runButton.disabled = true;
      setTimeout(function () {
        runButton.textContent = "Run Pipeline";
        runButton.disabled = false;
      }, 1500);
    });
  }

  function initRouteCharts(route) {
    switch (route) {
      case "dashboard":
        initDashboardCharts();
        break;

      default:
        break;
    }
  }

  window.PVPortalCharts = {
    bindTopbarActions: bindTopbarActions,
    initRouteCharts: initRouteCharts
  };
})();
