(function () {
  var USERS = [
    { id: 1, firstName: "Marcus", lastName: "Obi", email: "m.obi@pvlakehouse.io", role: "Admin", status: "Active", lastActive: "Just now", joined: "Jan 2023", scope: "Full Platform Access", notes: "Super admin" },
    { id: 2, firstName: "Aarav", lastName: "Kumar", email: "a.kumar@pvlakehouse.io", role: "ML Engineer", status: "Active", lastActive: "5 min ago", joined: "Mar 2023", scope: "Full Platform Access", notes: "" },
    { id: 3, firstName: "Sophie", lastName: "Vance", email: "s.vance@pvlakehouse.io", role: "Data Analyst", status: "Active", lastActive: "1 hour ago", joined: "Apr 2023", scope: "Dashboard and Reports Only", notes: "" },
    { id: 4, firstName: "Lena", lastName: "Fischer", email: "l.fischer@pvlakehouse.io", role: "ML Engineer", status: "Inactive", lastActive: "1 week ago", joined: "May 2023", scope: "ML Models Only", notes: "" },
    { id: 5, firstName: "Dario", lastName: "Conti", email: "d.conti@pvlakehouse.io", role: "Viewer", status: "Pending", lastActive: "Not yet", joined: "Jun 2023", scope: "Dashboard and Reports Only", notes: "Invite pending" },
    { id: 6, firstName: "Yuna", lastName: "Park", email: "y.park@pvlakehouse.io", role: "Data Analyst", status: "Suspended", lastActive: "2 months ago", joined: "Jul 2023", scope: "Pipeline Read-Only", notes: "Policy review" },
    { id: 7, firstName: "Priya", lastName: "Nair", email: "p.nair@pvlakehouse.io", role: "ML Engineer", status: "Active", lastActive: "4 hours ago", joined: "Nov 2023", scope: "ML Models Only", notes: "" },
    { id: 8, firstName: "Hana", lastName: "Suzuki", email: "h.suzuki@pvlakehouse.io", role: "Admin", status: "Active", lastActive: "20 min ago", joined: "Feb 2024", scope: "Full Platform Access", notes: "Co-admin" }
  ];

  var state = {
    filtered: USERS.slice(),
    page: 1,
    perPage: 8,
    sortKey: "name",
    sortDirection: "asc",
    editingId: null,
    deletingId: null
  };

  function getName(user) {
    return user.firstName + " " + user.lastName;
  }

  function getInitials(user) {
    return String(user.firstName || "").charAt(0).toUpperCase() + String(user.lastName || "").charAt(0).toUpperCase();
  }

  function roleClass(role) {
    if (role === "Admin") return "badge-role-admin";
    if (role === "ML Engineer") return "badge-role-ml";
    if (role === "Data Analyst") return "badge-role-analyst";
    return "badge-role-viewer";
  }

  function statusClass(status) {
    if (status === "Active") return "status-active";
    if (status === "Inactive") return "status-inactive";
    if (status === "Suspended") return "status-suspended";
    return "status-pending";
  }

  function updateStats() {
    var total = USERS.length;
    var active = USERS.filter(function (u) { return u.status === "Active"; }).length;
    var pending = USERS.filter(function (u) { return u.status === "Pending"; }).length;
    var suspended = USERS.filter(function (u) { return u.status === "Suspended"; }).length;

    document.getElementById("stat-total").textContent = String(total);
    document.getElementById("stat-total-meta").textContent = total === 1 ? "1 user in system" : total + " users in system";

    document.getElementById("stat-active").textContent = String(active);
    document.getElementById("stat-active-meta").textContent = (Math.round((active / Math.max(total, 1)) * 100)) + " percent of total";

    document.getElementById("stat-pending").textContent = String(pending);
    document.getElementById("stat-pending-meta").textContent = pending === 0 ? "No pending invite" : pending + " invite(s) waiting";

    document.getElementById("stat-suspended").textContent = String(suspended);
    document.getElementById("stat-suspended-meta").textContent = suspended === 0 ? "No suspended account" : suspended + " account(s) require review";
  }

  function compare(a, b) {
    var aValue = state.sortKey === "name" ? getName(a).toLowerCase() : String(a[state.sortKey] || "").toLowerCase();
    var bValue = state.sortKey === "name" ? getName(b).toLowerCase() : String(b[state.sortKey] || "").toLowerCase();

    if (aValue === bValue) return 0;
    if (state.sortDirection === "asc") {
      return aValue < bValue ? -1 : 1;
    }
    return aValue > bValue ? -1 : 1;
  }

  function applyFilters() {
    var query = (document.getElementById("account-search").value || "").trim().toLowerCase();
    var role = document.getElementById("account-role-filter").value;
    var status = document.getElementById("account-status-filter").value;

    state.filtered = USERS.filter(function (user) {
      var haystack = (getName(user) + " " + user.email + " " + user.role).toLowerCase();
      return (!query || haystack.indexOf(query) >= 0)
        && (!role || user.role === role)
        && (!status || user.status === status);
    }).sort(compare);

    state.page = 1;
    renderTable();
  }

  function renderTable() {
    var tbody = document.getElementById("accounts-tbody");
    var start = (state.page - 1) * state.perPage;
    var pageRows = state.filtered.slice(start, start + state.perPage);

    tbody.innerHTML = pageRows.map(function (user) {
      return ""
        + "<tr>"
        + "<td><input type=\"checkbox\" class=\"row-selector\" data-id=\"" + user.id + "\"></td>"
        + "<td><div class=\"user-cell\"><div class=\"user-avatar\">" + getInitials(user) + "</div><div><p class=\"user-name\">" + getName(user) + "</p><p class=\"user-email\">" + user.email + "</p></div></div></td>"
        + "<td><span class=\"badge " + roleClass(user.role) + "\">" + user.role + "</span></td>"
        + "<td><span class=\"status-label " + statusClass(user.status) + "\">" + user.status + "</span></td>"
        + "<td>" + user.lastActive + "</td>"
        + "<td>" + user.joined + "</td>"
        + "<td><div class=\"row-actions\"><button data-action=\"edit\" data-id=\"" + user.id + "\">Edit</button><button data-action=\"toggle-status\" data-id=\"" + user.id + "\">Toggle</button><button data-action=\"delete\" data-id=\"" + user.id + "\">Delete</button></div></td>"
        + "</tr>";
    }).join("");

    document.getElementById("accounts-count").textContent = state.filtered.length + (state.filtered.length === 1 ? " user" : " users");

    var end = Math.min(start + state.perPage, state.filtered.length);
    document.getElementById("accounts-pagination-info").textContent = "Showing " + (state.filtered.length ? (start + 1) : 0) + " to " + end + " of " + state.filtered.length + " users";

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
    }, 1800);
  }

  function resetForm() {
    document.getElementById("accounts-form").reset();
    document.getElementById("field-role").value = "ML Engineer";
    document.getElementById("field-status").value = "Active";
    document.getElementById("field-scope").value = "Full Platform Access";
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
  }

  function handleInviteClick() {
    state.editingId = null;
    document.getElementById("accounts-form-title").textContent = "Invite User";
    document.getElementById("accounts-form-subtitle").textContent = "Create a new account invitation";
    document.getElementById("accounts-submit-form-btn").textContent = "Send Invitation";
    resetForm();
    openOverlay("accounts-form-modal");
  }

  function openEditForm(id) {
    var user = USERS.find(function (item) { return item.id === id; });
    if (!user) return;

    state.editingId = id;
    document.getElementById("accounts-form-title").textContent = "Edit Account";
    document.getElementById("accounts-form-subtitle").textContent = "Update user profile and access";
    document.getElementById("accounts-submit-form-btn").textContent = "Save Changes";

    document.getElementById("field-first-name").value = user.firstName;
    document.getElementById("field-last-name").value = user.lastName;
    document.getElementById("field-email").value = user.email;
    document.getElementById("field-role").value = user.role;
    document.getElementById("field-status").value = user.status;
    document.getElementById("field-scope").value = user.scope;
    document.getElementById("field-notes").value = user.notes;

    openOverlay("accounts-form-modal");
  }

  function openDeleteForm(id) {
    var user = USERS.find(function (item) { return item.id === id; });
    if (!user) return;

    state.deletingId = id;
    document.getElementById("accounts-delete-target").textContent = getName(user) + " (" + user.email + ")";
    openOverlay("accounts-delete-modal");
  }

  function saveForm() {
    var firstName = document.getElementById("field-first-name").value.trim();
    var lastName = document.getElementById("field-last-name").value.trim();
    var email = document.getElementById("field-email").value.trim();
    var role = document.getElementById("field-role").value;
    var status = document.getElementById("field-status").value;
    var scope = document.getElementById("field-scope").value;
    var notes = document.getElementById("field-notes").value.trim();

    if (!firstName || !lastName || !email || email.indexOf("@") < 0) {
      showToast("Please fill required fields with a valid email");
      return;
    }

    if (state.editingId) {
      USERS = USERS.map(function (user) {
        if (user.id !== state.editingId) return user;
        return {
          id: user.id,
          firstName: firstName,
          lastName: lastName,
          email: email,
          role: role,
          status: status,
          lastActive: user.lastActive,
          joined: user.joined,
          scope: scope,
          notes: notes
        };
      });
      showToast("Account updated successfully");
    } else {
      USERS.unshift({
        id: Date.now(),
        firstName: firstName,
        lastName: lastName,
        email: email,
        role: role,
        status: status,
        lastActive: "Pending",
        joined: "Apr 2026",
        scope: scope,
        notes: notes
      });
      showToast("Invitation created successfully");
    }

    closeOverlay();
    applyFilters();
    updateStats();
  }

  function confirmDelete() {
    if (!state.deletingId) return;
    USERS = USERS.filter(function (user) { return user.id !== state.deletingId; });
    state.deletingId = null;
    closeOverlay();
    applyFilters();
    updateStats();
    showToast("Account deleted");
  }

  function toggleStatus(id) {
    USERS = USERS.map(function (user) {
      if (user.id !== id) return user;
      var nextStatus = user.status === "Suspended" ? "Active" : "Suspended";
      return {
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        role: user.role,
        status: nextStatus,
        lastActive: user.lastActive,
        joined: user.joined,
        scope: user.scope,
        notes: user.notes
      };
    });
    applyFilters();
    updateStats();
    showToast("Account status changed");
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
        document.querySelectorAll(".sort-btn").forEach(function (btn) { btn.classList.remove("active"); });
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
      if (!target || target.tagName !== "BUTTON") return;

      var id = Number(target.dataset.id);
      var action = target.dataset.action;
      if (action === "edit") {
        openEditForm(id);
      } else if (action === "delete") {
        openDeleteForm(id);
      } else if (action === "toggle-status") {
        toggleStatus(id);
      }
    });

    document.getElementById("invite-user-btn").addEventListener("click", handleInviteClick);
    document.getElementById("export-audit-btn").addEventListener("click", function () {
      showToast("Audit log export queued");
    });

    document.getElementById("accounts-close-form-modal").addEventListener("click", closeOverlay);
    document.getElementById("accounts-cancel-form-btn").addEventListener("click", closeOverlay);
    document.getElementById("accounts-submit-form-btn").addEventListener("click", saveForm);

    document.getElementById("accounts-close-delete-modal").addEventListener("click", closeOverlay);
    document.getElementById("accounts-cancel-delete-btn").addEventListener("click", closeOverlay);
    document.getElementById("accounts-confirm-delete-btn").addEventListener("click", confirmDelete);

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
    applyFilters();
    updateStats();
  });
})();
