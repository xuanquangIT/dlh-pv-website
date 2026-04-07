(function () {
  var ROLE_LABELS = {
    admin: "Admin",
    data_engineer: "Data Engineer",
    ml_engineer: "ML Engineer",
    analyst: "Data Analyst",
    system: "System"
  };

  var users = [];

  var state = {
    filtered: [],
    page: 1,
    perPage: 8,
    sortKey: "name",
    sortDirection: "asc",
    selectedUserId: null
  };

  function roleLabel(roleId) {
    return ROLE_LABELS[roleId] || String(roleId || "").replace(/_/g, " ");
  }

  function formatDate(value) {
    if (!value) {
      return "-";
    }

    var date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "-";
    }

    return date.toLocaleDateString("en-US", {
      month: "short",
      year: "numeric"
    });
  }

  function getName(user) {
    var fullName = String(user.full_name || "").trim();
    if (fullName) {
      return fullName;
    }
    return String(user.username || "");
  }

  function getInitials(user) {
    var name = getName(user);
    var parts = name.split(/\s+/).filter(Boolean).slice(0, 2);
    var initials = parts.map(function (part) {
      return part.charAt(0).toUpperCase();
    }).join("");

    return initials || "U";
  }

  function userStatus(user) {
    return user.is_active ? "Active" : "Inactive";
  }

  function roleClass(role) {
    if (role === "Admin") return "badge-role-admin";
    if (role === "Data Engineer") return "badge-role-engineer";
    if (role === "ML Engineer") return "badge-role-ml";
    return "badge-role-analyst";
  }

  function statusClass(status) {
    if (status === "Active") return "status-active";
    if (status === "Inactive") return "status-inactive";
    return "status-pending";
  }

  function findUserById(userId) {
    return users.find(function (item) {
      return item.id === userId;
    }) || null;
  }

  function compare(a, b) {
    var left = "";
    var right = "";

    if (state.sortKey === "name") {
      left = getName(a).toLowerCase();
      right = getName(b).toLowerCase();
    } else if (state.sortKey === "role") {
      left = roleLabel(a.role_id).toLowerCase();
      right = roleLabel(b.role_id).toLowerCase();
    } else if (state.sortKey === "status") {
      left = userStatus(a).toLowerCase();
      right = userStatus(b).toLowerCase();
    } else if (state.sortKey === "joined") {
      left = String(a.created_at || "").toLowerCase();
      right = String(b.created_at || "").toLowerCase();
    }

    if (left === right) return 0;
    if (state.sortDirection === "asc") {
      return left < right ? -1 : 1;
    }
    return left > right ? -1 : 1;
  }

  function updateStats() {
    var total = users.length;
    var active = users.filter(function (u) { return u.is_active; }).length;
    var adminCount = users.filter(function (u) { return u.role_id === "admin"; }).length;
    var inactive = users.filter(function (u) { return !u.is_active; }).length;

    document.getElementById("stat-total").textContent = String(total);
    document.getElementById("stat-total-meta").textContent =
      total === 1 ? "1 user in system" : total + " users in system";

    document.getElementById("stat-active").textContent = String(active);
    document.getElementById("stat-active-meta").textContent =
      Math.round((active / Math.max(total, 1)) * 100) + " percent of total";

    document.getElementById("stat-admin").textContent = String(adminCount);
    document.getElementById("stat-admin-meta").textContent =
      adminCount === 0 ? "No admin account" : adminCount + " admin account(s)";

    document.getElementById("stat-inactive").textContent = String(inactive);
    document.getElementById("stat-inactive-meta").textContent =
      inactive === 0 ? "No inactive account" : inactive + " inactive account(s)";
  }

  function applyFilters() {
    var query = (document.getElementById("account-search").value || "").trim().toLowerCase();
    var role = document.getElementById("account-role-filter").value;
    var status = document.getElementById("account-status-filter").value;

    state.filtered = users.filter(function (user) {
      var name = getName(user);
      var roleName = roleLabel(user.role_id);
      var userStatusText = userStatus(user);
      var haystack = (name + " " + user.email + " " + roleName).toLowerCase();

      return (!query || haystack.indexOf(query) >= 0)
        && (!role || roleName === role)
        && (!status || userStatusText === status);
    }).sort(compare);

    state.page = 1;
    renderTable();
  }

  function renderTable() {
    var tbody = document.getElementById("accounts-tbody");
    var start = (state.page - 1) * state.perPage;
    var pageRows = state.filtered.slice(start, start + state.perPage);

    if (!pageRows.length) {
      tbody.innerHTML = ""
        + "<tr>"
        + "<td colspan=\"6\" style=\"text-align:center; padding:24px;\">No accounts found</td>"
        + "</tr>";
    } else {
      tbody.innerHTML = pageRows.map(function (user) {
        var name = getName(user);
        var roleName = roleLabel(user.role_id);
        var status = userStatus(user);

        return ""
          + "<tr>"
          + "<td><div class=\"user-cell\"><div class=\"user-avatar\">" + getInitials(user)
          + "</div><div><p class=\"user-name\">" + name + "</p><p class=\"user-email\">"
          + user.email + "</p></div></div></td>"
          + "<td><span class=\"badge " + roleClass(roleName) + "\">" + roleName + "</span></td>"
          + "<td><span class=\"status-label " + statusClass(status) + "\">" + status + "</span></td>"
          + "<td>-</td>"
          + "<td>" + formatDate(user.created_at) + "</td>"
          + "<td><div class=\"row-actions\">"
          + "<button type=\"button\" data-action=\"toggle\" data-user-id=\"" + user.id + "\">"
          + (user.is_active ? "Set Inactive" : "Set Active")
          + "</button>"
          + "<button type=\"button\" data-action=\"password\" data-user-id=\"" + user.id + "\">Reset Password</button>"
          + "</div></td>"
          + "</tr>";
      }).join("");
    }

    document.getElementById("accounts-count").textContent =
      state.filtered.length + (state.filtered.length === 1 ? " user" : " users");

    var end = Math.min(start + state.perPage, state.filtered.length);
    document.getElementById("accounts-pagination-info").textContent =
      "Showing " + (state.filtered.length ? (start + 1) : 0)
      + " to " + end + " of " + state.filtered.length + " users";

    var maxPage = Math.max(1, Math.ceil(state.filtered.length / state.perPage));
    document.getElementById("accounts-prev-page").disabled = state.page <= 1;
    document.getElementById("accounts-next-page").disabled = state.page >= maxPage;
  }

  function showToast(message) {
    var toast = document.getElementById("accounts-toast");
    toast.textContent = message;
    toast.classList.add("show");
    window.setTimeout(function () {
      toast.classList.remove("show");
    }, 2000);
  }

  function openOverlay(modalId) {
    var overlay = document.getElementById("accounts-modal-overlay");
    overlay.classList.add("open");
    overlay.setAttribute("aria-hidden", "false");
    document.querySelectorAll(".accounts-modal").forEach(function (modal) {
      modal.classList.remove("open");
    });
    document.getElementById(modalId).classList.add("open");
  }

  function closeOverlay() {
    var overlay = document.getElementById("accounts-modal-overlay");
    overlay.classList.remove("open");
    overlay.setAttribute("aria-hidden", "true");
    state.selectedUserId = null;
  }

  function apiRequest(url, method, payload) {
    return fetch(url, {
      method: method,
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json"
      },
      credentials: "same-origin",
      body: payload ? JSON.stringify(payload) : undefined
    }).then(function (response) {
      if (!response.ok) {
        return response.text().then(function (message) {
          throw new Error(message || "Request failed");
        });
      }
      if (response.status === 204) {
        return null;
      }
      return response.json();
    });
  }

  function loadUsers() {
    return fetch("/auth/users", {
      method: "GET",
      headers: {
        "Accept": "application/json"
      },
      credentials: "same-origin"
    }).then(function (response) {
      if (!response.ok) {
        throw new Error("Failed to load account data");
      }
      return response.json();
    });
  }

  function refreshUsers() {
    return loadUsers().then(function (result) {
      users = Array.isArray(result) ? result : [];
      updateStats();
      applyFilters();
    });
  }

  function showLoadError() {
    var tbody = document.getElementById("accounts-tbody");
    tbody.innerHTML = ""
      + "<tr>"
      + "<td colspan=\"6\" style=\"text-align:center; padding:24px;\">"
      + "Unable to load account data"
      + "</td>"
      + "</tr>";
  }

  function onCreateAccount() {
    var form = document.getElementById("accounts-create-form");
    form.reset();
    document.getElementById("field-role").value = "data_engineer";
    document.getElementById("field-is-active").value = "true";
    openOverlay("accounts-create-modal");
  }

  function submitCreateAccount() {
    var username = document.getElementById("field-username").value.trim();
    var fullName = document.getElementById("field-full-name").value.trim();
    var email = document.getElementById("field-email").value.trim();
    var roleId = document.getElementById("field-role").value;
    var password = document.getElementById("field-password").value;
    var isActive = document.getElementById("field-is-active").value === "true";

    if (!username || !email || !password) {
      showToast("Username, email, and password are required");
      return;
    }
    if (password.length < 8) {
      showToast("Password must be at least 8 characters");
      return;
    }

    apiRequest("/auth/users", "POST", {
      username: username,
      full_name: fullName || null,
      email: email,
      role_id: roleId,
      password: password,
      is_active: isActive
    }).then(function () {
      closeOverlay();
      return refreshUsers();
    }).then(function () {
      showToast("Account created successfully");
    }).catch(function () {
      showToast("Failed to create account");
    });
  }

  function openPasswordModal(userId) {
    var user = findUserById(userId);
    if (!user) {
      return;
    }
    state.selectedUserId = user.id;
    document.getElementById("accounts-password-subtitle").textContent =
      getName(user) + " (" + user.username + ")";
    document.getElementById("field-new-password").value = "";
    openOverlay("accounts-password-modal");
  }

  function submitPasswordReset() {
    var newPassword = document.getElementById("field-new-password").value;
    if (!state.selectedUserId) {
      return;
    }
    if (newPassword.length < 8) {
      showToast("Password must be at least 8 characters");
      return;
    }

    apiRequest("/auth/users/" + state.selectedUserId + "/password", "PATCH", {
      new_password: newPassword
    }).then(function () {
      closeOverlay();
      showToast("Password updated");
    }).catch(function () {
      showToast("Failed to update password");
    });
  }

  function toggleUserStatus(userId) {
    var user = findUserById(userId);
    if (!user) {
      return;
    }

    apiRequest("/auth/users/" + user.id + "/status", "PATCH", {
      is_active: !user.is_active
    }).then(function () {
      return refreshUsers();
    }).then(function () {
      showToast("Account status updated");
    }).catch(function () {
      showToast("Failed to update account status");
    });
  }

  function bindEvents() {
    document.getElementById("account-search").addEventListener("input", applyFilters);
    document.getElementById("account-role-filter").addEventListener("change", applyFilters);
    document.getElementById("account-status-filter").addEventListener("change", applyFilters);

    document.querySelectorAll(".sort-btn").forEach(function (button) {
      button.addEventListener("click", function () {
        var key = button.dataset.sortKey;
        if (state.sortKey === key) {
          state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
        } else {
          state.sortKey = key;
          state.sortDirection = "asc";
        }
        document.querySelectorAll(".sort-btn").forEach(function (btn) {
          btn.classList.remove("active");
        });
        button.classList.add("active");
        applyFilters();
      });
    });

    document.getElementById("accounts-prev-page").addEventListener("click", function () {
      if (state.page > 1) {
        state.page -= 1;
        renderTable();
      }
    });

    document.getElementById("accounts-next-page").addEventListener("click", function () {
      var maxPage = Math.max(1, Math.ceil(state.filtered.length / state.perPage));
      if (state.page < maxPage) {
        state.page += 1;
        renderTable();
      }
    });

    document.getElementById("accounts-tbody").addEventListener("click", function (event) {
      var target = event.target;
      if (!target || target.tagName !== "BUTTON") {
        return;
      }

      var action = target.getAttribute("data-action");
      var userId = target.getAttribute("data-user-id");
      if (!action || !userId) {
        return;
      }

      if (action === "toggle") {
        toggleUserStatus(userId);
      } else if (action === "password") {
        openPasswordModal(userId);
      }
    });

    document.getElementById("create-account-btn").addEventListener("click", onCreateAccount);
    document.getElementById("accounts-create-submit-btn").addEventListener("click", submitCreateAccount);
    document.getElementById("accounts-password-submit-btn").addEventListener("click", submitPasswordReset);

    document.querySelectorAll("[data-close-overlay='true']").forEach(function (button) {
      button.addEventListener("click", closeOverlay);
    });

    document.getElementById("accounts-modal-overlay").addEventListener("click", function (event) {
      if (event.target.id === "accounts-modal-overlay") {
        closeOverlay();
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        closeOverlay();
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (window.PV_ROUTE !== "accounts") {
      return;
    }

    bindEvents();

    refreshUsers().catch(function () {
      users = [];
      updateStats();
      applyFilters();
      showLoadError();
    });
  });
})();
