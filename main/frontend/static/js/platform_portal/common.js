(function () {
  function applyTheme(theme) {
    var finalTheme = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", finalTheme);
    localStorage.setItem("pv-theme", finalTheme);

    var toggleBtn = document.getElementById("theme-toggle-btn");
    if (toggleBtn) {
      var isDark = finalTheme === "dark";
      toggleBtn.textContent = isDark ? "Light Mode" : "Dark Mode";
      toggleBtn.setAttribute("aria-pressed", isDark ? "true" : "false");
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    var initialTheme = localStorage.getItem("pv-theme");
    if (initialTheme !== "dark" && initialTheme !== "light") {
      initialTheme = "light";
    }
    applyTheme(initialTheme);

    var themeToggleBtn = document.getElementById("theme-toggle-btn");
    if (themeToggleBtn) {
      themeToggleBtn.addEventListener("click", function () {
        var currentTheme = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
        applyTheme(currentTheme === "dark" ? "light" : "dark");
      });
    }

    if (window.PVChatbotWidget && typeof window.PVChatbotWidget.initChatbot === "function") {
      window.PVChatbotWidget.initChatbot();
    }

    if (window.PVPortalCharts && typeof window.PVPortalCharts.bindTopbarActions === "function") {
      window.PVPortalCharts.bindTopbarActions();
    }
    if (window.PVPortalCharts && typeof window.PVPortalCharts.initRouteCharts === "function") {
      window.PVPortalCharts.initRouteCharts(window.PV_ROUTE);
    }

    if (window.PV_ROUTE === "solar_chat" && window.PVSolarChatPage && typeof window.PVSolarChatPage.initSolarChatPage === "function") {
      window.PVSolarChatPage.initSolarChatPage();
    }
  });
})();
