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
