/**
 * model_picker.js — Provider/model picker for Solar AI Chat composer.
 *
 * Each profile groups one provider (base_url + api_key + wire format) with
 * a list of selectable models. The dropdown renders models as <optgroup>
 * children of their parent profile so the user picks a (profile, model)
 * pair in one click.
 *
 * Surfaced ONLY for users whose backend role allows the picker
 * (admin / ml_engineer). For everyone else the picker stays hidden — the
 * server falls back to its startup default profile.
 *
 * Selection shape returned by getSelection():
 *   { model_profile_id: "<id>", model_name: "<model>" }
 *   or { model_profile_id: "", model_name: "" } when "Server default" picked.
 *
 * Exposes window.SolarModelPicker:
 *   - mount(container): attach the picker (idempotent)
 *   - getSelection():   current selection
 *   - refresh():        re-fetch from /solar-ai-chat/llm-profiles
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "solarChatModelPicker_v2";
  var roots = [];
  var profileList = [];
  var defaultProfileId = "";
  var defaultModelName = "";
  var loaded = false;
  var loadingPromise = null;

  function loadStored() {
    try {
      var raw = global.localStorage && global.localStorage.getItem(STORAGE_KEY);
      if (!raw) return { profile_id: "", model_name: "" };
      var parsed = JSON.parse(raw);
      return {
        profile_id: typeof parsed.profile_id === "string" ? parsed.profile_id : "",
        model_name: typeof parsed.model_name === "string" ? parsed.model_name : "",
      };
    } catch (_) {
      return { profile_id: "", model_name: "" };
    }
  }
  function saveStored(state) {
    try {
      if (!state.profile_id && !state.model_name) {
        global.localStorage.removeItem(STORAGE_KEY);
      } else {
        global.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      }
    } catch (_) {}
  }

  var selection = loadStored();

  function isAllowedRole() {
    try {
      var apiRole = String(document.body && document.body.dataset && document.body.dataset.apiRole || "").toLowerCase();
      return apiRole === "admin" || apiRole === "ml_engineer";
    } catch (_) { return false; }
  }

  function escapeHtml(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function fetchProfiles() {
    if (loadingPromise) return loadingPromise;
    loadingPromise = fetch("/solar-ai-chat/llm-profiles", {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    })
      .then(function (resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.json();
      })
      .then(function (data) {
        profileList = (data && Array.isArray(data.profiles)) ? data.profiles : [];
        defaultProfileId = (data && data.default_profile_id) || "";
        defaultModelName = (data && data.default_model_name) || "";
        loaded = true;

        // Validate stored selection: profile must still be enabled, and the
        // model must still belong to it. Otherwise drop to server-default.
        if (selection.profile_id) {
          var match = profileList.find(function (p) { return p.id === selection.profile_id; });
          if (!match) {
            selection = { profile_id: "", model_name: "" };
            saveStored(selection);
          } else if (selection.model_name && match.models.indexOf(selection.model_name) === -1) {
            selection.model_name = match.primary_model || "";
            saveStored(selection);
          }
        }
        renderAll();
      })
      .catch(function (err) {
        if (global.console && console.warn) console.warn("[ModelPicker] load failed", err);
        profileList = [];
        loaded = true;
        renderAll();
      });
    return loadingPromise;
  }

  function getSelection() {
    return {
      model_profile_id: selection.profile_id || "",
      model_name: selection.model_name || "",
    };
  }

  function setSelection(profileId, modelName) {
    selection = {
      profile_id: profileId || "",
      model_name: modelName || "",
    };
    saveStored(selection);
    renderAll();
  }

  // Encode (profile_id, model_name) into one <option value> with a separator
  // unlikely to appear in IDs. Empty value = "use server default".
  var SEP = "::";

  function encodeOptionValue(profileId, modelName) {
    return profileId + SEP + modelName;
  }
  function decodeOptionValue(raw) {
    if (!raw) return { profile_id: "", model_name: "" };
    var idx = raw.indexOf(SEP);
    if (idx === -1) return { profile_id: raw, model_name: "" };
    return {
      profile_id: raw.slice(0, idx),
      model_name: raw.slice(idx + SEP.length),
    };
  }

  function buildOptions() {
    var html = "";
    var defaultLabel = defaultProfileId
      ? "Server default — " + (function () {
          var p = profileList.find(function (x) { return x.id === defaultProfileId; });
          return p ? p.label + (defaultModelName ? " · " + defaultModelName : "") : defaultProfileId;
        })()
      : "Server default";
    var defaultSelected = (!selection.profile_id) ? " selected" : "";
    html += '<option value=""' + defaultSelected + ">" + escapeHtml(defaultLabel) + "</option>";

    profileList.forEach(function (p) {
      html += '<optgroup label="' + escapeHtml(p.label) + '">';
      p.models.forEach(function (m) {
        var sel = (selection.profile_id === p.id && selection.model_name === m) ? " selected" : "";
        var primaryTag = (m === p.primary_model) ? "  ★" : "";
        html += '<option value="' + escapeHtml(encodeOptionValue(p.id, m)) + '"' + sel + ">" +
                escapeHtml(m + primaryTag) +
                "</option>";
      });
      html += "</optgroup>";
    });
    return html;
  }

  function renderRoot(root) {
    if (!root) return;
    if (!isAllowedRole() || !loaded || profileList.length === 0) {
      root.innerHTML = "";
      root.classList.add("smp-hidden");
      return;
    }
    root.classList.remove("smp-hidden");

    root.innerHTML =
      '<svg class="smp-icon" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">' +
        '<path fill="currentColor" d="M12 2a5 5 0 0 0-5 5v1H5a3 3 0 0 0-3 3v8a3 3 0 0 0 3 3h14a3 3 0 0 0 3-3v-8a3 3 0 0 0-3-3h-2V7a5 5 0 0 0-5-5zm-3 6V7a3 3 0 1 1 6 0v1H9zm3 4a2 2 0 1 1 0 4 2 2 0 0 1 0-4z"/>' +
      "</svg>" +
      '<select class="smp-select" aria-label="LLM provider and model" data-tooltip="Pick provider · model">' +
        buildOptions() +
      "</select>";

    var select = root.querySelector(".smp-select");
    if (select) {
      select.addEventListener("change", function (e) {
        var decoded = decodeOptionValue(e.target.value);
        setSelection(decoded.profile_id, decoded.model_name);
      });
    }
  }

  function renderAll() { roots.forEach(renderRoot); }

  function mount(container) {
    if (!container) return;
    if (container.querySelector(":scope > .smp-toolbar")) return;
    var root = document.createElement("div");
    root.className = "smp-toolbar smp-toolbar-inline smp-hidden";
    container.appendChild(root);
    roots.push(root);
    if (loaded) renderRoot(root);
    else fetchProfiles();
  }

  function autoMount() {
    if (!isAllowedRole()) return;
    var slot = document.getElementById("solar-chat-model-picker-slot");
    if (slot) mount(slot);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", autoMount);
  } else {
    autoMount();
  }

  global.SolarModelPicker = {
    mount: mount,
    getSelection: getSelection,
    refresh: function () {
      loadingPromise = null;
      loaded = false;
      return fetchProfiles();
    },
  };
})(window);
