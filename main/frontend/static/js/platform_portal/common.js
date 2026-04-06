(function () {
  function getSessionUser() {
    try {
      var raw = sessionStorage.getItem("pv_user");
      if (!raw) {
        return null;
      }
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") {
        return null;
      }
      return parsed;
    } catch (error) {
      return null;
    }
  }

  function applySessionUser() {
    var user = getSessionUser();
    if (!user) {
      return false;
    }

    var userNameElement = document.querySelector(".user-name");
    if (userNameElement && user.name) {
      userNameElement.textContent = user.name;
    }

    var userRoleElement = document.querySelector(".user-role");
    if (userRoleElement && user.role) {
      userRoleElement.textContent = user.role;
    }

    var userInitialsElement = document.querySelector(".user-av");
    if (userInitialsElement && user.initials) {
      userInitialsElement.textContent = user.initials;
    }

    var modalNameElement = document.getElementById("logout-modal-name");
    if (modalNameElement && user.name) {
      modalNameElement.textContent = user.name;
    }

    var modalRoleElement = document.getElementById("logout-modal-role");
    if (modalRoleElement && user.role) {
      modalRoleElement.textContent = user.role;
    }

    var modalInitialsElement = document.getElementById("logout-modal-initials");
    if (modalInitialsElement && user.initials) {
      modalInitialsElement.textContent = user.initials;
    }

    return true;
  }

  function bindLogoutModal() {
    var modal = document.getElementById("logout-modal");
    var openButton = document.getElementById("open-logout-modal-btn");
    var cancelButton = document.getElementById("logout-modal-cancel");
    var confirmButton = document.getElementById("logout-modal-confirm");

    if (!modal || !openButton || !cancelButton || !confirmButton) {
      return;
    }

    function openModal() {
      modal.classList.add("open");
      modal.setAttribute("aria-hidden", "false");
    }

    function closeModal() {
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
    }

    openButton.addEventListener("click", openModal);
    cancelButton.addEventListener("click", closeModal);

    confirmButton.addEventListener("click", function () {
      sessionStorage.removeItem("pv_user");
      closeModal();
      window.location.assign("/auth/logout");
    });

    modal.addEventListener("click", function (event) {
      if (event.target === modal) {
        closeModal();
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && modal.classList.contains("open")) {
        closeModal();
      }
    });
  }

  function enforcePortalAuth() {
    if (!window.PV_ROUTE) {
      return;
    }

    // Backend auth is cookie-based (HttpOnly), so client JS cannot read it reliably.
    // Keep sessionStorage only as optional display data and let server guard routes.
    applySessionUser();
  }

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
    enforcePortalAuth();

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

    bindLogoutModal();
  });
})();
