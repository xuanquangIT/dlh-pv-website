(function () {
  const SOLAR_WELCOME_MESSAGE =
    "Xin chào! Tôi là **Solar AI**, trợ lý thông minh của PV Lakehouse.\n\n" +
    "Tôi có thể giúp bạn phân tích dữ liệu năng lượng, kiểm tra pipeline, xem metrics mô hình, hoặc giải thích các dự báo. Bạn cần hỗ trợ gì?";
  const PROJECT_STORAGE_KEY = "pv_solar_chat_projects";
  const PROJECT_SESSION_MAP_STORAGE_KEY = "pv_solar_chat_project_session_map";
  const ACTIVE_PROJECT_STORAGE_KEY = "pv_solar_chat_active_project";
  const NO_PROJECT_KEY = "__no_project__";
  const NO_PROJECT_LABEL = "No Project";

  function normalizeChatRole(role) {
    const normalized = String(role || "").trim().toLowerCase();
    if (normalized === "analyst") {
      return "data_analyst";
    }
    if (normalized === "data engineer") {
      return "data_engineer";
    }
    if (normalized === "ml engineer") {
      return "ml_engineer";
    }
    if (normalized === "data_engineer" || normalized === "ml_engineer" || normalized === "data_analyst" || normalized === "admin") {
      return normalized;
    }
    return "data_engineer";
  }

  function getActiveRole() {
    return normalizeChatRole(window.PV_CHAT_ROLE || window.PV_USER_ROLE || "");
  }

  function setActiveRole() {
    // Role is controlled by authenticated backend session and is immutable on UI.
  }

  const SolarChatApi = {
    async createSession(title) {
      return requestJson("/solar-ai-chat/sessions", {
        method: "POST",
        body: JSON.stringify({ title: title })
      });
    },

    async query(payload) {
      return requestJson("/solar-ai-chat/query", {
        method: "POST",
        body: JSON.stringify(payload)
      });
    },

    async getSession(sessionId) {
      return requestJson("/solar-ai-chat/sessions/" + encodeURIComponent(sessionId), {
        method: "GET"
      });
    },

    async listSessions() {
      return requestJson("/solar-ai-chat/sessions", {
        method: "GET"
      });
    },

    async updateSessionTitle(sessionId, title) {
      return requestJson("/solar-ai-chat/sessions/" + encodeURIComponent(sessionId) + "/rename", {
        method: "POST",
        body: JSON.stringify({ title: title })
      });
    },

    async deleteSession(sessionId) {
      return requestJson("/solar-ai-chat/sessions/" + encodeURIComponent(sessionId), {
        method: "DELETE"
      });
    }
  };

  function formatConversationGroup(isoText) {
    const date = new Date(isoText);
    if (Number.isNaN(date.getTime())) {
      return "Previous 7 days";
    }

    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const dateStart = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    const diffDays = Math.floor((todayStart.getTime() - dateStart.getTime()) / 86400000);

    if (diffDays <= 0) {
      return "Today";
    }
    if (diffDays === 1) {
      return "Yesterday";
    }
    return "Previous 7 days";
  }

  function sanitizeSessionTitle(message) {
    const compact = (message || "").replace(/\s+/g, " ").trim();
    if (!compact) {
      return "New conversation";
    }
    const maxLength = 56;
    return compact.length > maxLength ? compact.slice(0, maxLength).trim() + "..." : compact;
  }

  function getStoredProjects() {
    try {
      const raw = localStorage.getItem(PROJECT_STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.filter(function (name) {
        return Boolean(name) && name !== NO_PROJECT_KEY;
      }) : [];
    } catch (error) {
      return [];
    }
  }

  function saveStoredProjects(projects) {
    try {
      localStorage.setItem(PROJECT_STORAGE_KEY, JSON.stringify(projects));
    } catch (error) {
      // Ignore storage write failures in private mode.
    }
  }

  function getStoredProjectSessionMap() {
    try {
      const raw = localStorage.getItem(PROJECT_SESSION_MAP_STORAGE_KEY);
      if (!raw) {
        return {};
      }
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (error) {
      return {};
    }
  }

  function saveStoredProjectSessionMap(projectSessionMap) {
    try {
      localStorage.setItem(PROJECT_SESSION_MAP_STORAGE_KEY, JSON.stringify(projectSessionMap));
    } catch (error) {
      // Ignore storage write failures in private mode.
    }
  }

  function getStoredActiveProject(projects) {
    try {
      const storedValue = localStorage.getItem(ACTIVE_PROJECT_STORAGE_KEY) || "";
      if (storedValue === NO_PROJECT_KEY) {
        return NO_PROJECT_KEY;
      }
      if (storedValue && projects.some(function (projectName) { return projectName === storedValue; })) {
        return storedValue;
      }
    } catch (error) {
      // Ignore storage read failures in private mode.
    }
    return NO_PROJECT_KEY;
  }

  function normalizeApiErrorDetail(detail) {
    if (!detail) {
      return "Request failed";
    }
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      const parts = detail.map(function (item) {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object") {
          const msg = item.msg || item.message || "Validation error";
          const loc = Array.isArray(item.loc) ? item.loc.join(".") : "";
          return loc ? msg + " (" + loc + ")" : msg;
        }
        return String(item);
      }).filter(Boolean);
      return parts.length ? parts.join("; ") : "Request failed";
    }
    if (typeof detail === "object") {
      return detail.message || JSON.stringify(detail);
    }
    return String(detail);
  }

  function saveStoredActiveProject(projectName) {
    try {
      localStorage.setItem(ACTIVE_PROJECT_STORAGE_KEY, projectName);
    } catch (error) {
      // Ignore storage write failures in private mode.
    }
  }

  async function requestJson(url, options) {
    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json"
      },
      ...options
    });

    if (!response.ok) {
      let detailMessage = "Request failed";
      try {
        const errorData = await response.json();
        detailMessage = normalizeApiErrorDetail(errorData && errorData.detail);
      } catch (e) {
        detailMessage = response.statusText || detailMessage;
      }
      throw new Error(detailMessage);
    }

    if (response.status === 204) {
      return null;
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.toLowerCase().includes("application/json")) {
      return null;
    }

    return response.json();
  }

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatAssistantContent(content) {
    return escapeHtml(content)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  function createWelcomeMessage() {
    return {
      role: "assistant",
      content: SOLAR_WELCOME_MESSAGE,
      timestamp: new Date().toISOString(),
      isIntro: true
    };
  }

  function withWelcomeMessage(messages) {
    const hasWelcome = messages.some(function (message) {
      return Boolean(message.isIntro);
    });
    if (hasWelcome) {
      return messages;
    }
    return [createWelcomeMessage(), ...messages];
  }

  function formatTime(timestamp) {
    if (!timestamp) {
      const now = new Date();
      return now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    }
    const date = new Date(timestamp);
    if (Number.isNaN(date.getTime())) {
      return "--:--";
    }
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }

  function createMessageElement(role, content, timestamp) {
    const normalizedRole = role === "user" ? "user" : "assistant";
    const row = document.createElement("div");
    row.className = "msg-row " + (normalizedRole === "user" ? "msg-row-user" : "msg-row-assistant");

    const message = document.createElement("div");
    message.className = "msg " + (normalizedRole === "user" ? "msg-user" : "msg-bot");
    if (normalizedRole === "assistant") {
      message.innerHTML = formatAssistantContent(content);
    } else {
      message.textContent = content;
    }

    const time = document.createElement("div");
    time.className = "msg-time";
    time.textContent = formatTime(timestamp);

    row.appendChild(message);
    row.appendChild(time);
    return row;
  }

  function MessageList(container) {
    this.container = container;
  }

  MessageList.prototype.render = function (messages) {
    this.container.innerHTML = "";
    messages.forEach(function (message) {
      const role = message.role === "assistant" || message.role === "bot" ? "assistant" : "user";
      const content = message.content || "";
      this.container.appendChild(createMessageElement(role, content, message.timestamp));
    }, this);
    this.scrollToBottom();
  };

  MessageList.prototype.append = function (role, content, timestamp) {
    this.container.appendChild(createMessageElement(role, content, timestamp));
    this.scrollToBottom();
  };

  MessageList.prototype.scrollToBottom = function () {
    this.container.scrollTop = this.container.scrollHeight;
  };

  function MessageInput(textarea, sendButton, onSend) {
    this.textarea = textarea;
    this.sendButton = sendButton;
    this.onSend = onSend;
  }

  MessageInput.prototype.init = function () {
    this.sendButton.addEventListener("click", () => {
      this.submit();
    });

    this.textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.submit();
      }
    });
  };

  MessageInput.prototype.submit = function () {
    const value = this.textarea.value.trim();
    if (!value) {
      return;
    }
    this.onSend(value);
  };

  MessageInput.prototype.setDisabled = function (disabled) {
    this.textarea.disabled = disabled;
    this.sendButton.disabled = disabled;
  };

  MessageInput.prototype.clear = function () {
    this.textarea.value = "";
  };

  function initSolarChatPage() {
    const messagesElement = document.getElementById("page-chat-messages");
    const inputElement = document.getElementById("page-chat-input");
    const sendButton = document.getElementById("page-chat-send");
    const statusElement = document.getElementById("solar-chat-status-text");
    const errorElement = document.getElementById("page-chat-error");
    const loadingElement = document.getElementById("page-chat-loading");
    const exportButton = document.getElementById("solar-chat-export-btn");
    const pipelineButton = document.getElementById("pipeline-status-btn");
    const newChatButton = document.getElementById("solar-chat-new-chat-btn");
    const sessionListElement = document.getElementById("solar-chat-session-list");
    const searchInputElement = document.getElementById("solar-chat-search-input");
    const projectListElement = document.getElementById("solar-chat-project-list");
    const newProjectButton = document.getElementById("solar-chat-new-project-btn");
    const projectModalElement = document.getElementById("solar-chat-project-modal");
    const projectNameInputElement = document.getElementById("solar-chat-project-name-input");
    const projectCreateButton = document.getElementById("solar-chat-project-create-btn");
    const projectCancelButton = document.getElementById("solar-chat-project-cancel-btn");
    const projectModalErrorElement = document.getElementById("solar-chat-project-modal-error");
    const deleteProjectModalElement = document.getElementById("solar-chat-delete-project-modal");
    const deleteProjectTargetElement = document.getElementById("solar-chat-delete-project-target");
    const deleteProjectCancelButton = document.getElementById("solar-chat-delete-project-cancel-btn");
    const deleteProjectConfirmButton = document.getElementById("solar-chat-delete-project-confirm-btn");
    const renameModalElement = document.getElementById("solar-chat-rename-modal");
    const renameModalTitleElement = document.getElementById("solar-chat-rename-modal-title");
    const renameInputElement = document.getElementById("solar-chat-rename-input");
    const renameErrorElement = document.getElementById("solar-chat-rename-error");
    const renameCancelButton = document.getElementById("solar-chat-rename-cancel-btn");
    const renameConfirmButton = document.getElementById("solar-chat-rename-confirm-btn");
    const deleteSessionModalElement = document.getElementById("solar-chat-delete-session-modal");
    const deleteSessionTargetElement = document.getElementById("solar-chat-delete-session-target");
    const deleteSessionCancelButton = document.getElementById("solar-chat-delete-session-cancel-btn");
    const deleteSessionConfirmButton = document.getElementById("solar-chat-delete-session-confirm-btn");
    const sessionMenuElement = document.getElementById("solar-chat-session-menu");
    const sessionMenuEditButton = document.getElementById("solar-chat-session-menu-edit");
    const sessionMenuDeleteButton = document.getElementById("solar-chat-session-menu-delete");

    if (
      !messagesElement ||
      !inputElement ||
      !sendButton ||
      !statusElement ||
      !errorElement ||
      !loadingElement ||
      !exportButton ||
      !pipelineButton ||
      !newChatButton ||
      !sessionListElement ||
      !searchInputElement ||
      !projectListElement ||
      !newProjectButton ||
      !projectModalElement ||
      !projectNameInputElement ||
      !projectCreateButton ||
      !projectCancelButton ||
      !projectModalErrorElement ||
      !deleteProjectModalElement ||
      !deleteProjectTargetElement ||
      !deleteProjectCancelButton ||
      !deleteProjectConfirmButton ||
      !renameModalElement ||
      !renameModalTitleElement ||
      !renameInputElement ||
      !renameErrorElement ||
      !renameCancelButton ||
      !renameConfirmButton ||
      !deleteSessionModalElement ||
      !deleteSessionTargetElement ||
      !deleteSessionCancelButton ||
      !deleteSessionConfirmButton ||
      !sessionMenuElement ||
      !sessionMenuEditButton ||
      !sessionMenuDeleteButton
    ) {
      return;
    }

    const createProjectBackdrop = projectModalElement.querySelector(".solar-chat-modal-backdrop");
    const deleteProjectBackdrop = deleteProjectModalElement.querySelector(".solar-chat-modal-backdrop");
    const renameBackdrop = renameModalElement.querySelector(".solar-chat-modal-backdrop");
    const deleteSessionBackdrop = deleteSessionModalElement.querySelector(".solar-chat-modal-backdrop");
    if (!createProjectBackdrop || !deleteProjectBackdrop || !renameBackdrop || !deleteSessionBackdrop) {
      return;
    }

    const initialProjects = getStoredProjects();
    const initialActiveProject = getStoredActiveProject(initialProjects);
    const state = {
      role: getActiveRole(),
      sessionId: "",
      messages: [createWelcomeMessage()],
      loading: false,
      modelUsed: "",
      sessions: [],
      filteredSessions: [],
      projects: initialProjects,
      sessionQuery: "",
      projectSessionMap: getStoredProjectSessionMap(),
      activeProject: initialActiveProject,
      pendingDeleteProject: "",
      pendingDeleteSessionId: "",
      renameTarget: null,
      contextSessionId: ""
    };

    const messageList = new MessageList(messagesElement);
    const messageInput = new MessageInput(inputElement, sendButton, async (message) => {
      await sendMessageFlow(message);
    });
    const modelElement = document.getElementById("ctx-model");
    const topKElement = document.getElementById("ctx-top-k");
    const roleElement = document.getElementById("ctx-role");
    const sessionElement = document.getElementById("ctx-session-id");
    const messageCountElement = document.getElementById("solar-chat-message-count");

    messageInput.init();
    messageList.render(state.messages);

    function getProjectLabel(projectKey) {
      return projectKey === NO_PROJECT_KEY ? NO_PROJECT_LABEL : projectKey;
    }

    function getSessionProject(sessionId) {
      return state.projectSessionMap[sessionId] || NO_PROJECT_KEY;
    }

    function setLoading(loading, messageText) {
      state.loading = loading;
      messageInput.setDisabled(loading);
      loadingElement.hidden = !loading;
      if (loading) {
        loadingElement.textContent = messageText || "Assistant is analyzing your request.";
      }
    }

    function setStatus(text) {
      statusElement.textContent = text;
    }

    function updateContext() {
      if (roleElement) {
        roleElement.textContent = state.role;
      }
      if (sessionElement) {
        sessionElement.textContent = state.sessionId || "not-created";
      }
      if (modelElement && state.modelUsed) {
        modelElement.textContent = state.modelUsed;
      }
      if (topKElement && !topKElement.textContent) {
        topKElement.textContent = "5";
      }
      if (messageCountElement) {
        messageCountElement.textContent = String(state.messages.length);
      }
    }

    function setError(message) {
      if (!message) {
        errorElement.hidden = true;
        errorElement.textContent = "";
        return;
      }
      errorElement.hidden = false;
      errorElement.textContent = message;
    }

    function openProjectModal() {
      projectModalErrorElement.hidden = true;
      projectModalErrorElement.textContent = "";
      projectNameInputElement.value = "";
      projectModalElement.hidden = false;
      projectNameInputElement.focus();
    }

    function closeProjectModal() {
      projectModalElement.hidden = true;
      projectModalErrorElement.hidden = true;
      projectModalErrorElement.textContent = "";
    }

    function openDeleteProjectModal(projectName) {
      state.pendingDeleteProject = projectName;
      deleteProjectTargetElement.textContent = "Project: " + projectName;
      deleteProjectModalElement.hidden = false;
    }

    function closeDeleteProjectModal() {
      deleteProjectModalElement.hidden = true;
      state.pendingDeleteProject = "";
    }

    function openDeleteSessionModal(sessionSummary) {
      state.pendingDeleteSessionId = sessionSummary.session_id;
      deleteSessionTargetElement.textContent = "Conversation: " + (sessionSummary.title || "New conversation");
      deleteSessionModalElement.hidden = false;
      closeSessionMenu();
    }

    function closeDeleteSessionModal() {
      deleteSessionModalElement.hidden = true;
      state.pendingDeleteSessionId = "";
    }

    function closeSessionMenu() {
      sessionMenuElement.hidden = true;
      state.contextSessionId = "";
    }

    function openSessionMenu(sessionId, event) {
      state.contextSessionId = sessionId;
      sessionMenuElement.style.left = event.clientX + "px";
      sessionMenuElement.style.top = event.clientY + "px";
      sessionMenuElement.hidden = false;
    }

    function getSessionById(sessionId) {
      return state.sessions.find(function (summary) {
        return summary.session_id === sessionId;
      }) || null;
    }

    function openRenameModal(target) {
      state.renameTarget = target;
      renameErrorElement.hidden = true;
      renameErrorElement.textContent = "";
      renameModalTitleElement.textContent = target.type === "project" ? "Rename Project" : "Rename Conversation";
      renameInputElement.value = target.currentName || "";
      renameModalElement.hidden = false;
      renameInputElement.focus();
      renameInputElement.select();
      closeSessionMenu();
    }

    function closeRenameModal() {
      renameModalElement.hidden = true;
      renameErrorElement.hidden = true;
      renameErrorElement.textContent = "";
      state.renameTarget = null;
    }

    function ensureProjectSessionMap(sessions) {
      let changed = false;

      sessions.forEach(function (summary) {
        if (!state.projectSessionMap[summary.session_id]) {
          state.projectSessionMap[summary.session_id] = NO_PROJECT_KEY;
          changed = true;
        }
      });

      Object.keys(state.projectSessionMap).forEach(function (sessionId) {
        const stillExists = sessions.some(function (summary) {
          return summary.session_id === sessionId;
        });
        if (!stillExists) {
          delete state.projectSessionMap[sessionId];
          changed = true;
          return;
        }

        const key = state.projectSessionMap[sessionId];
        if (key !== NO_PROJECT_KEY && !state.projects.includes(key)) {
          state.projectSessionMap[sessionId] = NO_PROJECT_KEY;
          changed = true;
        }
      });

      if (changed) {
        saveStoredProjectSessionMap(state.projectSessionMap);
      }
    }

    function setActiveProject(projectKey) {
      state.activeProject = projectKey;
      saveStoredActiveProject(projectKey);
      renderProjects();
      applySessionFilter(state.sessionQuery);

      if (state.sessionId && getSessionProject(state.sessionId) !== projectKey) {
        resetConversation();
      }
    }

    function renderProjects() {
      projectListElement.innerHTML = "";

      const unassignedRow = document.createElement("div");
      unassignedRow.className = "solar-chat-project-row";
      const unassignedButton = document.createElement("button");
      unassignedButton.type = "button";
      unassignedButton.className = "solar-chat-project-item" + (state.activeProject === NO_PROJECT_KEY ? " active" : "");
      unassignedButton.textContent = NO_PROJECT_LABEL;
      unassignedButton.addEventListener("click", function () {
        setActiveProject(NO_PROJECT_KEY);
      });
      unassignedRow.appendChild(unassignedButton);
      projectListElement.appendChild(unassignedRow);

      state.projects.forEach(function (projectName) {
        const row = document.createElement("div");
        row.className = "solar-chat-project-row";

        const item = document.createElement("button");
        item.type = "button";
        item.className = "solar-chat-project-item" + (state.activeProject === projectName ? " active" : "");
        item.textContent = projectName;
        item.addEventListener("click", function () {
          setActiveProject(projectName);
        });

        const editButton = document.createElement("button");
        editButton.type = "button";
        editButton.className = "solar-chat-project-edit";
        editButton.textContent = "Edit";
        editButton.title = "Rename project";
        editButton.addEventListener("click", function (event) {
          event.stopPropagation();
          openRenameModal({ type: "project", projectName: projectName, currentName: projectName });
        });

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "solar-chat-project-delete";
        deleteButton.textContent = "Delete";
        deleteButton.title = "Delete project";
        deleteButton.addEventListener("click", function (event) {
          event.stopPropagation();
          openDeleteProjectModal(projectName);
        });

        row.appendChild(item);
        row.appendChild(editButton);
        row.appendChild(deleteButton);
        projectListElement.appendChild(row);
      });
    }

    function deleteProject(projectName) {
      state.projects = state.projects.filter(function (name) {
        return name !== projectName;
      });

      Object.keys(state.projectSessionMap).forEach(function (sessionId) {
        if (state.projectSessionMap[sessionId] === projectName) {
          state.projectSessionMap[sessionId] = NO_PROJECT_KEY;
        }
      });

      if (state.activeProject === projectName) {
        state.activeProject = NO_PROJECT_KEY;
      }

      saveStoredProjects(state.projects);
      saveStoredProjectSessionMap(state.projectSessionMap);
      saveStoredActiveProject(state.activeProject);
      renderProjects();
      applySessionFilter(state.sessionQuery);
      setStatus("Project deleted");
      closeDeleteProjectModal();
    }

    function renameProject(oldName, newName) {
      if (oldName === newName) {
        closeRenameModal();
        return;
      }

      if (state.projects.some(function (projectName) { return projectName.toLowerCase() === newName.toLowerCase(); })) {
        renameErrorElement.hidden = false;
        renameErrorElement.textContent = "Project name already exists.";
        return;
      }

      state.projects = state.projects.map(function (projectName) {
        return projectName === oldName ? newName : projectName;
      });

      Object.keys(state.projectSessionMap).forEach(function (sessionId) {
        if (state.projectSessionMap[sessionId] === oldName) {
          state.projectSessionMap[sessionId] = newName;
        }
      });

      if (state.activeProject === oldName) {
        state.activeProject = newName;
      }

      saveStoredProjects(state.projects);
      saveStoredProjectSessionMap(state.projectSessionMap);
      saveStoredActiveProject(state.activeProject);
      renderProjects();
      applySessionFilter(state.sessionQuery);
      closeRenameModal();
      setStatus("Project renamed");
    }

    function getConversationTimeLabel(summary) {
      const updatedAt = new Date(summary.updated_at || "");
      if (Number.isNaN(updatedAt.getTime())) {
        return "Unknown";
      }
      return updatedAt.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    }

    function groupSessionsByDate(summaries) {
      return summaries.reduce(function (grouped, summary) {
        const groupKey = formatConversationGroup(summary.updated_at);
        if (!grouped[groupKey]) {
          grouped[groupKey] = [];
        }
        grouped[groupKey].push(summary);
        return grouped;
      }, {});
    }

    function renderSessionGroups() {
      sessionListElement.innerHTML = "";
      const groups = groupSessionsByDate(state.filteredSessions);
      const orderedKeys = ["Today", "Yesterday", "Previous 7 days"];

      const hasData = orderedKeys.some(function (key) {
        return groups[key] && groups[key].length;
      });

      if (!hasData) {
        const empty = document.createElement("div");
        empty.className = "solar-chat-conv-empty";
        empty.textContent = "No conversations found.";
        sessionListElement.appendChild(empty);
        return;
      }

      orderedKeys.forEach(function (groupKey) {
        const groupItems = groups[groupKey] || [];
        if (!groupItems.length) {
          return;
        }

        const heading = document.createElement("div");
        heading.className = "solar-chat-conv-group-title";
        heading.textContent = groupKey;
        sessionListElement.appendChild(heading);

        groupItems.forEach(function (summary) {
          const row = document.createElement("div");
          row.className = "solar-chat-conv-row";

          const item = document.createElement("button");
          item.type = "button";
          item.className = "solar-chat-conv-item" + (summary.session_id === state.sessionId ? " active" : "");

          const title = document.createElement("div");
          title.className = "solar-chat-conv-item-title";
          title.textContent = summary.title || "New conversation";

          const meta = document.createElement("div");
          meta.className = "solar-chat-conv-item-meta";
          meta.textContent = getConversationTimeLabel(summary) + " | " + String(summary.message_count || 0) + " messages";

          item.appendChild(title);
          item.appendChild(meta);
          item.addEventListener("click", async function () {
            await openSession(summary.session_id);
          });

          item.addEventListener("contextmenu", function (event) {
            event.preventDefault();
            openSessionMenu(summary.session_id, event);
          });

          row.appendChild(item);
          sessionListElement.appendChild(row);
        });
      });
    }

    function applySessionFilter(query) {
      const normalized = (query || "").trim().toLowerCase();
      state.sessionQuery = normalized;

      const projectScoped = state.sessions.filter(function (summary) {
        return getSessionProject(summary.session_id) === state.activeProject;
      });

      state.filteredSessions = normalized
        ? projectScoped.filter(function (summary) {
            return (summary.title || "").toLowerCase().includes(normalized);
          })
        : projectScoped;

      renderSessionGroups();
    }

    async function ensureSession(titleHint) {
      if (state.sessionId) {
        return state.sessionId;
      }
      const created = await SolarChatApi.createSession(sanitizeSessionTitle(titleHint));
      state.sessionId = created.session_id;
      state.projectSessionMap[state.sessionId] = state.activeProject;
      saveStoredProjectSessionMap(state.projectSessionMap);
      updateContext();
      await refreshSessionList();
      return state.sessionId;
    }

    async function refreshSessionList() {
      const sessions = await SolarChatApi.listSessions();
      state.sessions = sessions;
      ensureProjectSessionMap(state.sessions);
      applySessionFilter(state.sessionQuery);
    }

    async function loadSessionMessages(sessionId) {
      const detail = await SolarChatApi.getSession(sessionId);
      const loadedMessages = (detail.messages || []).map(function (message) {
        return {
          role: message.sender === "assistant" ? "assistant" : "user",
          content: message.content || "",
          timestamp: message.timestamp
        };
      });
      state.messages = withWelcomeMessage(loadedMessages);
      messageList.render(state.messages);
      updateContext();
    }

    async function openSession(sessionId) {
      if (!sessionId || state.loading) {
        return;
      }
      try {
        setError("");
        setLoading(true, "Loading selected conversation...");
        state.sessionId = sessionId;

        const mappedProject = getSessionProject(sessionId);
        if (mappedProject !== state.activeProject) {
          state.activeProject = mappedProject;
          saveStoredActiveProject(mappedProject);
          renderProjects();
          applySessionFilter(state.sessionQuery);
        }

        await loadSessionMessages(sessionId);
        renderSessionGroups();
        setStatus("Online · Ready to assist");
      } catch (error) {
        setStatus("Error");
        setError(error instanceof Error ? error.message : "Unable to open conversation.");
      } finally {
        setLoading(false);
      }
    }

    async function renameSession(sessionId, newName) {
      await SolarChatApi.updateSessionTitle(sessionId, newName);
      await refreshSessionList();

      if (state.sessionId === sessionId) {
        await loadSessionMessages(sessionId);
      }
    }

    async function deleteSession(sessionId) {
      await SolarChatApi.deleteSession(sessionId);
      delete state.projectSessionMap[sessionId];
      saveStoredProjectSessionMap(state.projectSessionMap);

      if (state.sessionId === sessionId) {
        await resetConversation();
      }

      await refreshSessionList();
    }

    async function sendMessageFlow(text) {
      if (state.loading) {
        return;
      }

      const userMessage = {
        role: "user",
        content: text,
        timestamp: new Date().toISOString()
      };

      state.messages.push(userMessage);
      messageList.render(state.messages);
      messageInput.clear();
      updateContext();

      try {
        setError("");
        setLoading(true, "Sending message to assistant...");
        setStatus("Processing");

        const sessionId = await ensureSession(text);

        const response = await SolarChatApi.query({
          message: text,
          session_id: sessionId
        });

        state.modelUsed = response.model_used || state.modelUsed;
        state.messages.push({
          role: "assistant",
          content: response.answer || "",
          timestamp: new Date().toISOString()
        });
        messageList.render(state.messages);
        updateContext();

        if (response.warning_message) {
          setError(response.warning_message);
        }

        refreshSessionList().catch(function () {
          // Keep UI responsive even if sidebar refresh fails.
        });
        setStatus("Ready");
      } catch (error) {
        setStatus("Error");
        setError(error instanceof Error ? error.message : "Unexpected error occurred.");
      } finally {
        setLoading(false);
      }
    }

    async function resetConversation() {
      state.sessionId = "";
      state.messages = [createWelcomeMessage()];
      state.modelUsed = "";
      messageList.render(state.messages);
      setError("");
      setStatus("Online · Ready to assist");
      updateContext();
      renderSessionGroups();
    }

    exportButton.addEventListener("click", function () {
      setStatus("Export is not implemented yet.");
    });

    pipelineButton.addEventListener("click", async function () {
      await sendMessageFlow("Show latest pipeline status");
    });

    newChatButton.addEventListener("click", async function () {
      await resetConversation();
      setStatus("New chat started");
    });

    searchInputElement.addEventListener("input", function () {
      applySessionFilter(searchInputElement.value);
    });

    newProjectButton.addEventListener("click", function () {
      openProjectModal();
    });

    projectCancelButton.addEventListener("click", function () {
      closeProjectModal();
    });

    createProjectBackdrop.addEventListener("click", function () {
      closeProjectModal();
    });

    projectCreateButton.addEventListener("click", function () {
      const cleaned = (projectNameInputElement.value || "").trim();
      if (!cleaned) {
        projectModalErrorElement.hidden = false;
        projectModalErrorElement.textContent = "Project name is required.";
        return;
      }

      if (state.projects.some(function (projectName) { return projectName.toLowerCase() === cleaned.toLowerCase(); })) {
        projectModalErrorElement.hidden = false;
        projectModalErrorElement.textContent = "Project name already exists.";
        return;
      }

      state.projects = [cleaned, ...state.projects];
      saveStoredProjects(state.projects);
      setActiveProject(cleaned);
      closeProjectModal();
      setStatus("Project created");
    });

    projectNameInputElement.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        closeProjectModal();
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        projectCreateButton.click();
      }
    });

    deleteProjectBackdrop.addEventListener("click", function () {
      closeDeleteProjectModal();
    });

    deleteProjectCancelButton.addEventListener("click", function () {
      closeDeleteProjectModal();
    });

    deleteProjectConfirmButton.addEventListener("click", function () {
      if (!state.pendingDeleteProject) {
        closeDeleteProjectModal();
        return;
      }
      deleteProject(state.pendingDeleteProject);
    });

    renameBackdrop.addEventListener("click", function () {
      closeRenameModal();
    });

    renameCancelButton.addEventListener("click", function () {
      closeRenameModal();
    });

    renameConfirmButton.addEventListener("click", async function () {
      if (!state.renameTarget) {
        closeRenameModal();
        return;
      }

      const cleaned = (renameInputElement.value || "").trim();
      if (!cleaned) {
        renameErrorElement.hidden = false;
        renameErrorElement.textContent = "Name is required.";
        return;
      }

      if (state.renameTarget.type === "project") {
        renameProject(state.renameTarget.projectName, cleaned);
        return;
      }

      try {
        await renameSession(state.renameTarget.sessionId, cleaned);
        closeRenameModal();
        setStatus("Conversation renamed");
      } catch (error) {
        renameErrorElement.hidden = false;
        renameErrorElement.textContent = error instanceof Error ? error.message : "Unable to rename conversation.";
      }
    });

    renameInputElement.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        closeRenameModal();
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        renameConfirmButton.click();
      }
    });

    deleteSessionBackdrop.addEventListener("click", function () {
      closeDeleteSessionModal();
    });

    deleteSessionCancelButton.addEventListener("click", function () {
      closeDeleteSessionModal();
    });

    deleteSessionConfirmButton.addEventListener("click", async function () {
      if (!state.pendingDeleteSessionId) {
        closeDeleteSessionModal();
        return;
      }

      try {
        await deleteSession(state.pendingDeleteSessionId);
        closeDeleteSessionModal();
        setStatus("Conversation deleted");
      } catch (error) {
        setStatus("Error");
        setError(error instanceof Error ? error.message : "Unable to delete conversation.");
      }
    });

    sessionMenuEditButton.addEventListener("click", function () {
      if (!state.contextSessionId) {
        closeSessionMenu();
        return;
      }
      const summary = getSessionById(state.contextSessionId);
      if (!summary) {
        closeSessionMenu();
        return;
      }
      openRenameModal({
        type: "session",
        sessionId: summary.session_id,
        currentName: summary.title || "New conversation"
      });
      closeSessionMenu();
    });

    sessionMenuDeleteButton.addEventListener("click", function () {
      if (!state.contextSessionId) {
        closeSessionMenu();
        return;
      }
      const summary = getSessionById(state.contextSessionId);
      if (!summary) {
        closeSessionMenu();
        return;
      }
      openDeleteSessionModal(summary);
      closeSessionMenu();
    });

    document.addEventListener("click", function (event) {
      if (sessionMenuElement.hidden) {
        return;
      }
      if (!sessionMenuElement.contains(event.target)) {
        closeSessionMenu();
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") {
        return;
      }
      if (!projectModalElement.hidden) {
        closeProjectModal();
      }
      if (!deleteProjectModalElement.hidden) {
        closeDeleteProjectModal();
      }
      if (!renameModalElement.hidden) {
        closeRenameModal();
      }
      if (!deleteSessionModalElement.hidden) {
        closeDeleteSessionModal();
      }
      if (!sessionMenuElement.hidden) {
        closeSessionMenu();
      }
    });

    document.querySelectorAll("[data-chat-prompt]").forEach(function (button) {
      button.addEventListener("click", async function () {
        const prompt = button.getAttribute("data-chat-prompt") || "";
        if (!prompt) {
          return;
        }
        await sendMessageFlow(prompt);
      });
    });

    renderProjects();
    updateContext();
    setStatus("Loading conversations");
    refreshSessionList()
      .then(function () {
        setStatus("Online · Ready to assist");
      })
      .catch(function (error) {
        setStatus("Error");
        setError(error instanceof Error ? error.message : "Failed to load conversations.");
      });
  }

  window.PVSolarChatPage = {
    initSolarChatPage: initSolarChatPage,
    getActiveRole: getActiveRole,
    setActiveRole: setActiveRole,
    SolarChatApi: SolarChatApi,
    createWelcomeMessage: createWelcomeMessage,
    createMessageElement: createMessageElement
  };
})();
