(function () {
  document.addEventListener("DOMContentLoaded", function () {
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
