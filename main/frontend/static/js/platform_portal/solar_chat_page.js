(function () {
  // Thinking-trace panel is an engineering affordance. The backend always
  // returns the trace (used for observability / bug reports) but the UI
  // only renders it for engineering roles or when ?debug=1 is set.
  function canSeeThinkingTrace() {
    try {
      var search = String(window.location.search || "");
      if (/(?:^|[?&])debug=1(?:&|$)/.test(search)) return true;
      var body = document.body;
      if (body && body.dataset) {
        if (body.dataset.isDevMode === "true") return true;
        var apiRole = String(body.dataset.apiRole || "").toLowerCase();
        if (apiRole === "admin" || apiRole === "ml_engineer" || apiRole === "data_engineer") {
          return true;
        }
      }
    } catch (_) { /* defensive */ }
    return false;
  }

  const SOLAR_WELCOME_MESSAGE =
    "Xin chào! Tôi là **Solar AI**, trợ lý thông minh của PV Lakehouse.\n\n" +
    "Tôi có thể giúp bạn phân tích dữ liệu năng lượng, kiểm tra pipeline, xem metrics mô hình, hoặc giải thích các dự báo. Bạn cần hỗ trợ gì?";
  const PROJECT_STORAGE_KEY = "pv_solar_chat_projects";
  const PROJECT_SESSION_MAP_STORAGE_KEY = "pv_solar_chat_project_session_map";
  const ACTIVE_PROJECT_STORAGE_KEY = "pv_solar_chat_active_project";
  const LAST_SESSION_STORAGE_KEY = "pv_solar_chat_last_session_id";

  function loadStoredLastSessionId() {
    try { return localStorage.getItem(LAST_SESSION_STORAGE_KEY) || ""; } catch (_) { return ""; }
  }
  function saveStoredLastSessionId(sessionId) {
    try {
      if (sessionId) localStorage.setItem(LAST_SESSION_STORAGE_KEY, sessionId);
      else localStorage.removeItem(LAST_SESSION_STORAGE_KEY);
    } catch (_) { /* ignore */ }
  }
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

    /**
     * Stream a query via SSE.
     * @param {object} payload - { message, session_id }
     * @param {object} handlers - { onStatus, onThinkingStep, onToolResult, onTextDelta, onDone, onError }
     * @returns {AbortController} - Call .abort() to cancel
     */
    queryStream(payload, handlers) {
      const controller = new AbortController();
      const {
        onStatus = () => {},
        onThinkingStep = () => {},
        onToolResult = () => {},
        onTextDelta = () => {},
        onDone = () => {},
        onError = () => {}
      } = handlers || {};

      (async () => {
        let response;
        try {
          response = await fetch("/solar-ai-chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: controller.signal
          });
        } catch (fetchErr) {
          if (fetchErr.name !== "AbortError") {
            onError({ message: fetchErr.message || "Network error" });
          }
          return;
        }

        if (!response.ok) {
          let detail = "Stream request failed";
          try { const d = await response.json(); detail = (d && d.detail) || detail; } catch (e) { /* ignore */ }
          onError({ message: detail });
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop();  // last partial chunk stays in buffer
            for (const block of lines) {
              const line = block.trim();
              if (!line.startsWith("data: ")) continue;
              let evt;
              try { evt = JSON.parse(line.slice(6)); } catch (e) { continue; }
              switch (evt.event) {
                case "status_update":  onStatus(evt); break;
                case "thinking_step": onThinkingStep(evt); break;
                case "tool_result":   onToolResult(evt); break;
                case "text_delta":    onTextDelta(evt); break;
                case "done":          onDone(evt); break;
                case "error":         onError(evt); break;
                default: break;
              }
            }
          }
        } catch (readErr) {
          if (readErr.name !== "AbortError") {
            onError({ message: readErr.message || "Stream read error" });
          }
        }
      })();

      return controller;
    },

    async getSession(sessionId) {
      return requestJson("/solar-ai-chat/sessions/" + encodeURIComponent(sessionId), {
        method: "GET"
      });
    },

    async listSessions(limit = 20, offset = 0) {
      return requestJson("/solar-ai-chat/sessions?limit=" + encodeURIComponent(limit) + "&offset=" + encodeURIComponent(offset), {
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
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fallbackAssistantFormatting(content) {
    return escapeHtml(content || "")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  function formatAssistantContent(content) {
    const markdown = String(content || "");
    const canParseMarkdown = Boolean(window.marked && typeof window.marked.parse === "function");
    const canSanitizeHtml = Boolean(window.DOMPurify && typeof window.DOMPurify.sanitize === "function");

    if (!canParseMarkdown || !canSanitizeHtml) {
      return fallbackAssistantFormatting(markdown);
    }

    const parsedHtml = window.marked.parse(markdown, {
      gfm: true,
      breaks: true,
      headerIds: false,
      mangle: false,
    });

    const safeHtml = window.DOMPurify.sanitize(parsedHtml, {
      USE_PROFILES: { html: true },
    });

    const wrapper = document.createElement("div");
    wrapper.innerHTML = safeHtml;
    wrapper.querySelectorAll("a[href]").forEach(function (anchor) {
      anchor.setAttribute("target", "_blank");
      anchor.setAttribute("rel", "noopener noreferrer");
    });
    return wrapper.innerHTML;
  }

  function buildTypingIndicatorHtml() {
    return [
      '<span class="typing-indicator" role="status" aria-label="Assistant is typing">',
      '<span class="typing-dot"></span>',
      '<span class="typing-dot"></span>',
      '<span class="typing-dot"></span>',
      "</span>"
    ].join("");
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

  function buildThinkingTraceHtml(thinkingTrace, isOpen) {
    if (!thinkingTrace || !thinkingTrace.steps || thinkingTrace.steps.length === 0) {
      return "";
    }
    // Feature-flag gate: SHOW_THINKING_TRACE.
    if (window.ToolStatusCard && !window.ToolStatusCard.canShowTrace()) {
      return "";
    }

    // Parse the summary — format is "N tool calls · Topic · model-name"
    const summary = String(thinkingTrace.summary || "");

    // Count total steps for the pill badge
    const stepCount = thinkingTrace.steps.length;
    const allOk = thinkingTrace.steps.every(function(s) {
      return s.status === "success" || s.status === "ok" || s.status === "skipped";
    });

    const toggleLabel = (isOpen ? "▾" : "▸") + " Thought for " + stepCount +
      " step" + (stepCount !== 1 ? "s" : "") +
      " <span class=\"msg-thinking-badge " + (allOk ? "ok" : "warn") + "\">" +
      (allOk ? "done" : "issues") + "</span>";

    const html = [
      '<details class="msg-thinking"' + (isOpen ? " open" : "") + ">",
      '<summary class="msg-thinking-toggle">' + toggleLabel + "</summary>",
      '<div class="msg-thinking-body">',
    ];

    if (summary) {
      html.push('<div class="msg-thinking-summary">' + escapeHtml(summary) + "</div>");
    }

    html.push('<div class="msg-thinking-list">');

    thinkingTrace.steps.forEach(function(step) {
      if (window.ToolStatusCard) {
        const rowHtml = window.ToolStatusCard.renderRow({
          step: step.step,
          tool_name: step.tool_name,
          label: step.step || step.tool_name || "",
          status: step.status,
          detail: step.detail
        });
        if (rowHtml) html.push(rowHtml);
        return;
      }
      // Fallback path (component not loaded): legacy renderer.
      const isOk = step.status === "success" || step.status === "ok";
      const isWarn = step.status === "warning" || step.status === "error";
      const isRunning = step.status === "running";
      let icon;
      if (isRunning) {
        icon = '<span class="task-tracker-spinner" aria-hidden="true"></span>';
      } else if (isOk) {
        icon = '<span class="task-tracker-icon ok" aria-hidden="true">✓</span>';
      } else if (isWarn) {
        icon = '<span class="task-tracker-icon error" aria-hidden="true">✕</span>';
      } else {
        icon = '<span class="task-tracker-icon skipped" aria-hidden="true">–</span>';
      }
      const rowClass = isRunning ? "task-tracker-row running" : "task-tracker-row";
      const label = step.step || step.tool_name || "";
      const detail = step.detail ? ' <span class="task-tracker-dur">' + escapeHtml(step.detail) + "</span>" : "";
      html.push(
        '<div class="' + rowClass + '">',
        "  " + icon,
        '  <span class="task-tracker-label">' + escapeHtml(label) + "</span>",
        detail,
        "</div>"
      );
    });

    html.push("</div></div></details>");
    return html.join("");
  }


  function renderMessageContent(messageElement, role, content, isPending, hasThinking) {
    if (role === "assistant") {
      if (isPending && !content) {
        messageElement.classList.add("msg-typing");
        messageElement.style.display = "block";
        messageElement.innerHTML = buildTypingIndicatorHtml();
      } else if (content) {
        messageElement.classList.remove("msg-typing");
        messageElement.style.display = "block";
        messageElement.innerHTML = formatAssistantContent(content);
      } else {
        // No content and not pending: hide the bubble if there is a thinking trace
        messageElement.style.display = hasThinking ? "none" : "block";
        messageElement.innerHTML = "";
      }
      return;
    }

    messageElement.classList.remove("msg-typing");
    messageElement.style.display = "block";
    messageElement.textContent = content || "";
  }

  function createMessageElement(role, content, timestamp, thinkingTrace, messageId, isPending) {
    const normalizedRole = role === "user" ? "user" : "assistant";
    const row = document.createElement("div");
    row.className = "msg-row " + (normalizedRole === "user" ? "msg-row-user" : "msg-row-assistant");
    if (messageId) {
      row.dataset.messageId = messageId;
    }

    if (thinkingTrace && normalizedRole === "assistant" && canSeeThinkingTrace()) {
      const thinkingWrapper = document.createElement("div");
      thinkingWrapper.className = "msg-thinking-wrapper";
      thinkingWrapper.innerHTML = buildThinkingTraceHtml(thinkingTrace, false);
      row.appendChild(thinkingWrapper);
    }

    const message = document.createElement("div");
    message.className = "msg " + (normalizedRole === "user" ? "msg-user" : "msg-bot");
    renderMessageContent(message, normalizedRole, content, Boolean(isPending), Boolean(thinkingTrace));

    const time = document.createElement("div");
    time.className = "msg-time";
    time.textContent = formatTime(timestamp);

    row.appendChild(message);
    row.appendChild(time);
    return row;
  }

  function updateMessageElement(row, role, content, timestamp, thinkingTrace, isPending) {
    const normalizedRole = role === "user" ? "user" : "assistant";

    let thinkingWrapper = row.querySelector(".msg-thinking-wrapper");
    if (thinkingTrace && normalizedRole === "assistant" && canSeeThinkingTrace()) {
      const details = thinkingWrapper ? thinkingWrapper.querySelector("details") : null;
      const isOpen = details ? details.open : false;
      const html = buildThinkingTraceHtml(thinkingTrace, isOpen);
      if (!thinkingWrapper) {
        thinkingWrapper = document.createElement("div");
        thinkingWrapper.className = "msg-thinking-wrapper";
        row.insertBefore(thinkingWrapper, row.firstChild);
      }
      thinkingWrapper.innerHTML = html;
    } else if (thinkingWrapper) {
      thinkingWrapper.remove();
    }

    const message = row.querySelector(".msg");
    if (message) {
      renderMessageContent(message, normalizedRole, content, Boolean(isPending), Boolean(thinkingTrace));
    }

    const time = row.querySelector(".msg-time");
    if (time) {
      time.textContent = formatTime(timestamp);
    }
  }

  function mountVizExtras(messageId, viz) {
    if (!viz) return;
    if (!viz.data_table && !viz.chart && !viz.kpi_cards) return;
    const row = document.querySelector('[data-message-id="' + messageId + '"]');
    if (!row) return;

    row.classList.add("msg-row-has-viz");
    let extras = row.querySelector(".solar-viz-extras");
    if (!extras) {
      extras = document.createElement("div");
      extras.className = "solar-viz-extras";
      const msg = row.querySelector(".msg");
      if (msg && msg.parentNode === row) {
        row.insertBefore(extras, msg.nextSibling);
      } else {
        row.appendChild(extras);
      }
    } else {
      extras.innerHTML = "";
    }

    if (viz.kpi_cards && window.KpiCards) {
      const kpiEl = document.createElement("div");
      kpiEl.className = "solar-kpi-cards";
      extras.appendChild(kpiEl);
      window.KpiCards.render(kpiEl, viz.kpi_cards);
    }
    if (viz.chart && window.ChartRenderer) {
      const chartEl = document.createElement("div");
      extras.appendChild(chartEl);
      window.ChartRenderer.render(chartEl, viz.chart);
    }
    if (viz.data_table && window.DataTable) {
      const tblEl = document.createElement("div");
      extras.appendChild(tblEl);
      window.DataTable.render(tblEl, viz.data_table);
    }
  }

  function MessageList(container) {
    this.container = container;
  }


  MessageList.prototype.render = function (messages) {
    this.container.innerHTML = "";
    messages.forEach(function (message) {
      const role = message.role === "assistant" || message.role === "bot" ? "assistant" : "user";
      const content = message.content || "";
      const thinkingTrace = message.thinkingTrace || message.thinking_trace || null;
      const messageId = typeof message.id === "string" ? message.id : "";
      const isPending = Boolean(message.isPending);
      this.container.appendChild(createMessageElement(role, content, message.timestamp, thinkingTrace, messageId, isPending));
    }, this);
    // Re-attach any viz payloads (KPI/chart/table) that were previously computed
    // so they survive a full re-render (e.g. after sending a follow-up message).
    messages.forEach(function (message) {
      if (message && message.viz && message.id) {
        try { mountVizExtras(message.id, message.viz); } catch (_) {}
      }
    });
    this.scrollToBottom();
  };

  MessageList.prototype.append = function (role, content, timestamp, thinkingTrace, messageId, isPending) {
    const row = createMessageElement(role, content, timestamp, thinkingTrace, messageId || "", Boolean(isPending));
    this.container.appendChild(row);
    this.scrollToBottom();
    return row;
  };

  MessageList.prototype.findById = function (messageId) {
    if (!messageId) {
      return null;
    }
    const rows = this.container.querySelectorAll(".msg-row");
    for (let index = 0; index < rows.length; index += 1) {
      if (rows[index].dataset.messageId === messageId) {
        return rows[index];
      }
    }
    return null;
  };

  MessageList.prototype.updateById = function (messageId, role, content, timestamp, thinkingTrace, isPending) {
    const row = this.findById(messageId);
    if (!row) {
      return false;
    }
    updateMessageElement(row, role, content, timestamp, thinkingTrace, Boolean(isPending));
    this.scrollToBottom();
    return true;
  };

  MessageList.prototype.removeById = function (messageId) {
    const row = this.findById(messageId);
    if (!row) {
      return false;
    }
    row.remove();
    return true;
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

  // Bug #9: allow pre-filling the textarea for chip/quick-action clicks
  MessageInput.prototype.setValue = function (text) {
    this.textarea.value = text;
  };

  function initSolarChatPage() {
    const messagesElement = document.getElementById("page-chat-messages");
    const inputElement = document.getElementById("page-chat-input");
    const sendButton = document.getElementById("page-chat-send");
    const statusElement = document.getElementById("solar-chat-status-text");
    const errorElement = document.getElementById("page-chat-error");
    const feedbackContainer = document.getElementById("page-chat-feedback");
    const suggestionsElement = document.getElementById("solar-chat-suggestions");
    const promptTriggerElement = document.getElementById("solar-chat-prompt-trigger");
    const suggestionsCloseElement = document.getElementById("solar-chat-suggestions-close");
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
      sessionsLimit: 20,
      sessionsOffset: 0,
      hasMoreSessions: true,
      loadingHistory: false,
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
      if (suggestionsElement) {
        const hasConversation = state.messages.length > 1;
        setSuggestionsVisible(!hasConversation);
      }
    }

    function setError(message) {
      if (!message) {
        errorElement.hidden = true;
        errorElement.textContent = "";
        if (feedbackContainer) feedbackContainer.hidden = true;
        return;
      }
      errorElement.hidden = false;
      errorElement.textContent = message;
      if (feedbackContainer) feedbackContainer.hidden = false;
    }

    function shouldSurfaceWarningAsError(warningMessage) {
      return false;
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
      const currentScrollTop = sessionListElement.scrollTop;
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

      // Restore scroll position after recreating DOM to prevent jumping during infinite scroll
      sessionListElement.scrollTop = currentScrollTop;
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
      saveStoredLastSessionId(state.sessionId);
      state.projectSessionMap[state.sessionId] = state.activeProject;
      saveStoredProjectSessionMap(state.projectSessionMap);
      updateContext();
      await refreshSessionList();
      return state.sessionId;
    }

    async function refreshSessionList(append = false) {
      if (!append) {
        state.sessionsOffset = 0;
        state.hasMoreSessions = true;
      }
      if (!state.hasMoreSessions) return;

      const newSessions = await SolarChatApi.listSessions(state.sessionsLimit, state.sessionsOffset);
      if (newSessions.length < state.sessionsLimit) {
        state.hasMoreSessions = false;
      }

      if (append) {
        state.sessions = state.sessions.concat(newSessions);
      } else {
        state.sessions = newSessions;
      }
      state.sessionsOffset += newSessions.length;
      
      ensureProjectSessionMap(state.sessions);
      applySessionFilter(state.sessionQuery);
    }

    async function loadSessionMessages(sessionId) {
      const detail = await SolarChatApi.getSession(sessionId);
      const loadedMessages = (detail.messages || []).map(function (message) {
        return {
          id: message.id || "",
          role: message.sender === "assistant" ? "assistant" : "user",
          content: message.content || "",
          timestamp: message.timestamp,
          thinkingTrace: message.thinking_trace || null,
          viz: (message.data_table || message.chart || message.kpi_cards) ? {
            data_table: message.data_table || null,
            chart: message.chart || null,
            kpi_cards: message.kpi_cards || null
          } : null
        };
      });
      state.messages = withWelcomeMessage(loadedMessages);
      messageList.render(state.messages);
      // Re-attach viz extras for any assistant messages that carry them.
      loadedMessages.forEach(function (m) {
        if (m.viz && m.id) {
          try { mountVizExtras(m.id, m.viz); } catch (_) {}
        }
      });
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
        saveStoredLastSessionId(sessionId);

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

    // ----------------------------------------------------------------
    // Live Task Tracker
    // ----------------------------------------------------------------

    function TaskTracker(container) {
      this.container = container;
      this._tasks = [];
      this._statusText = "Processing your request...";
      // We start open.
      this._isOpen = true;
    }

    TaskTracker.prototype._render = function () {
      var stepCount = this._tasks.length;
      var allDone = this._tasks.length > 0 && this._tasks.every(function(t) {
        return t.status !== "running";
      });
      var allOk = allDone && this._tasks.every(function(t) {
        return t.status === "ok" || t.status === "skipped";
      });

      var arrowChar = this._isOpen ? "▾" : "▸";
      var badge;
      if (!allDone) {
        badge = '<span class="msg-thinking-badge running">working</span>';
      } else if (allOk) {
        badge = '<span class="msg-thinking-badge ok">done</span>';
      } else {
        badge = '<span class="msg-thinking-badge warn">issues</span>';
      }

      var toggleLabel = arrowChar + " Thinking\u2026 " + stepCount +
        " step" + (stepCount !== 1 ? "s" : "") + " " + badge;

      var html = [
        '<details class="msg-thinking"' + (this._isOpen ? " open" : "") + ">",
        '<summary class="msg-thinking-toggle">' + toggleLabel + "</summary>",
        '<div class="msg-thinking-body">',
        '<div class="msg-thinking-summary">' + escapeHtml(this._statusText) + "</div>",
        '<div class="msg-thinking-list">'
      ];

      var tasksHtml = this._tasks.map(function (task) {
        if (window.ToolStatusCard) {
          return window.ToolStatusCard.renderRow(task);
        }
        // Fallback (component not loaded): legacy inline renderer.
        var iconHtml;
        var rowClass = "task-tracker-row";
        if (task.status === "running") {
          iconHtml = '<span class="task-tracker-spinner" aria-hidden="true"></span>';
          rowClass += " running";
        } else if (task.status === "ok") {
          iconHtml = '<span class="task-tracker-icon ok" aria-hidden="true">✓</span>';
        } else if (task.status === "error") {
          iconHtml = '<span class="task-tracker-icon error" aria-hidden="true">✕</span>';
        } else if (task.status === "denied") {
          iconHtml = '<span class="task-tracker-icon denied" aria-hidden="true">⊘</span>';
        } else {
          iconHtml = '<span class="task-tracker-icon skipped" aria-hidden="true">–</span>';
        }
        var durationHtml = task.duration_ms != null
          ? ' <span class="task-tracker-dur">' + task.duration_ms + "ms</span>"
          : "";
        return [
          '<div class="' + rowClass + '" aria-live="polite">',
          "  " + iconHtml,
          '  <span class="task-tracker-label">' + escapeHtml(task.label || task.tool_name) + "</span>",
          durationHtml,
          "</div>"
        ].join("");
      }).join("");

      html.push(tasksHtml);
      html.push("</div></div></details>");
      
      // Only set innerHTML if there's actually a task or status, otherwise keep it empty so it doesn't take space
      if (this._tasks.length > 0) {
        this.container.innerHTML = html.join("");
      } else {
        this.container.innerHTML = "";
      }
    };


    TaskTracker.prototype.setStatus = function(text) {
      this._statusText = text;
      this._render();
    };

    TaskTracker.prototype.setOpen = function(isOpen) {
      this._isOpen = isOpen;
      this._render();
    };

    TaskTracker.prototype.addRunning = function (step, toolName, label) {
      var idx = this._tasks.findIndex(function (t) { return t.step === step; });
      var task = { step: step, tool_name: toolName, label: label, status: "running", duration_ms: null };
      if (idx >= 0) {
        this._tasks[idx] = task;
      } else {
        this._tasks.push(task);
      }
      this._render();
    };

    TaskTracker.prototype.markDone = function (step, status, durationMs) {
      var idx = this._tasks.findIndex(function (t) { return t.step === step; });
      if (idx >= 0) {
        this._tasks[idx].status = status;
        this._tasks[idx].duration_ms = durationMs || 0;
        this._render();
      }
    };

    // ----------------------------------------------------------------
    // sendMessageFlow — streaming version
    // ----------------------------------------------------------------

    async function sendMessageFlow(text) {
      if (state.loading) {
        return;
      }

      const pendingMessageId =
        "pending-" + Date.now().toString(36) + "-" + Math.random().toString(16).slice(2, 8);
      const userMessage = {
        role: "user",
        content: text,
        timestamp: new Date().toISOString()
      };
      const pendingAssistantMessage = {
        id: pendingMessageId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        isPending: true
      };

      state.messages.push(userMessage);
      state.messages.push(pendingAssistantMessage);
      messageList.render(state.messages);
      messageInput.clear();
      updateContext();

      // Find the row and replace its typing indicator with our TaskTracker container and a content container
      const pendingRow = messageList.findById(pendingMessageId);
      let taskTracker = null;
      let textContentEl = null;

      if (pendingRow) {
        const trackerContainer = document.createElement("div");
        trackerContainer.className = "msg-thinking-wrapper";
        // task tracker starts empty, we don't render until events come in, or we just render an empty state?
        // Actually TaskTracker _render creates the HTML inside trackerContainer.
        pendingRow.insertBefore(trackerContainer, pendingRow.firstChild);
        taskTracker = new TaskTracker(trackerContainer);
        
        // We do not call _render yet because tasks is empty, so it would show "0 steps".
        // But the typing indicator is currently inside msgEl, displaying beautifully!
        textContentEl = pendingRow.querySelector(".msg");
      }

      let streamController = null;

      try {
        setError("");
        setLoading(true, "Connecting to Solar AI…");
        setStatus("Processing");

        const sessionId = await ensureSession(text);

        await new Promise(function (resolve, reject) {
          const toolSelection = (window.SolarToolPicker && window.SolarToolPicker.getSelection())
            || { tool_mode: "auto", allowed_tools: null, tool_hints: [] };
          streamController = SolarChatApi.queryStream(
            {
              message: text,
              session_id: sessionId,
              tool_mode: toolSelection.tool_mode || "auto",
              allowed_tools: toolSelection.allowed_tools || null,
              tool_hints: toolSelection.tool_hints || [],
            },
            {
              onStatus: function (evt) {
                setStatus(evt.text || "Processing");
                if (taskTracker) taskTracker.setStatus(evt.text || "Processing");
              },
              onThinkingStep: function (evt) {
                if (taskTracker) taskTracker.addRunning(evt.step, evt.tool_name, evt.label || evt.tool_name);
              },
              onToolResult: function (evt) {
                if (taskTracker) taskTracker.markDone(evt.step, evt.status || "ok", evt.duration_ms);
              },
              onTextDelta: function (evt) {
                if (textContentEl) {
                  if (textContentEl.classList.contains("msg-typing")) {
                    textContentEl.innerHTML = "";
                    textContentEl.classList.remove("msg-typing");
                  }
                  textContentEl.textContent = (textContentEl.textContent || "") + (evt.delta || "");
                }
              },
              onDone: function (evt) {
                if (window.ToolStatusCard && evt.ui_features) {
                  window.ToolStatusCard.updateFeatures(evt.ui_features);
                }
                if (taskTracker) taskTracker.setOpen(false); // Collapse when done

                state.modelUsed = evt.model_used || state.modelUsed;
                var vizPayload = (evt.data_table || evt.chart || evt.kpi_cards) ? {
                  data_table: evt.data_table || null,
                  chart: evt.chart || null,
                  kpi_cards: evt.kpi_cards || null
                } : null;
                var assistantMessage = {
                  id: pendingMessageId,
                  role: "assistant",
                  content: evt.answer || "",
                  timestamp: new Date().toISOString(),
                  thinkingTrace: evt.thinking_trace || null,
                  viz: vizPayload
                };
                var pendingIndex = state.messages.findIndex(function (item) {
                  return item.id === pendingMessageId && item.isPending;
                });
                if (pendingIndex >= 0) {
                  state.messages.splice(pendingIndex, 1, assistantMessage);
                  if (!messageList.updateById(
                    pendingMessageId,
                    "assistant",
                    assistantMessage.content,
                    assistantMessage.timestamp,
                    assistantMessage.thinkingTrace,
                    false
                  )) {
                    messageList.render(state.messages);
                  }
                } else {
                  state.messages.push(assistantMessage);
                  messageList.append(
                    "assistant",
                    assistantMessage.content,
                    assistantMessage.timestamp,
                    assistantMessage.thinkingTrace,
                    "",
                    false
                  );
                }
                updateContext();

                if (shouldSurfaceWarningAsError(evt.warning_message)) {
                  setError(evt.warning_message);
                }

                try {
                  mountVizExtras(pendingMessageId, {
                    data_table: evt.data_table || null,
                    chart: evt.chart || null,
                    kpi_cards: evt.kpi_cards || null
                  });
                } catch (vizErr) {
                  if (window.console && console.warn) console.warn("viz mount failed", vizErr);
                }

                refreshSessionList().catch(function () {});
                setStatus("Ready");
                resolve();
              },
              onError: function (evt) {
                reject(new Error(evt.message || "Stream error"));
              }
            }
          );
        });

      } catch (error) {
        const pendingIndex = state.messages.findIndex(function (item) {
          return item.id === pendingMessageId && item.isPending;
        });
        if (pendingIndex >= 0) {
          state.messages.splice(pendingIndex, 1);
          if (!messageList.removeById(pendingMessageId)) {
            messageList.render(state.messages);
          }
        }
        setStatus("Error");
        setError(error instanceof Error ? error.message : "Unexpected error occurred.");
      } finally {
        setLoading(false);
      }
    }

    async function resetConversation() {
      state.sessionId = "";
      saveStoredLastSessionId("");
      state.messages = [createWelcomeMessage()];
      state.modelUsed = "";
      messageList.render(state.messages);
      setError("");
      setStatus("Online · Ready to assist");
      updateContext();
      renderSessionGroups();
    }

    if (exportButton) {
      exportButton.addEventListener("click", function () {
        setStatus("Export is not implemented yet.");
      });
    }

    // Right-drawer history toggle
    const historyToggle = document.getElementById("solar-chat-history-toggle");
    const historyDrawer = document.getElementById("solar-chat-history-drawer");
    const historyScrim = document.getElementById("solar-chat-drawer-scrim");
    const historyClose = document.getElementById("solar-chat-drawer-close");
    function openHistoryDrawer() {
      if (!historyDrawer) return;
      historyDrawer.hidden = false;
      historyDrawer.classList.add("is-open");
      if (historyScrim) { historyScrim.hidden = false; historyScrim.classList.add("is-visible"); }
      if (historyToggle) historyToggle.setAttribute("aria-expanded", "true");
    }
    function closeHistoryDrawer() {
      if (!historyDrawer) return;
      historyDrawer.classList.remove("is-open");
      if (historyScrim) historyScrim.classList.remove("is-visible");
      if (historyToggle) historyToggle.setAttribute("aria-expanded", "false");
      window.setTimeout(function () {
        if (!historyDrawer.classList.contains("is-open")) {
          historyDrawer.hidden = true;
          if (historyScrim) historyScrim.hidden = true;
        }
      }, 260);
    }
    if (historyToggle) {
      historyToggle.addEventListener("click", function () {
        if (historyDrawer && historyDrawer.classList.contains("is-open")) {
          closeHistoryDrawer();
        } else {
          openHistoryDrawer();
        }
      });
    }
    if (historyClose) historyClose.addEventListener("click", closeHistoryDrawer);
    if (historyScrim) historyScrim.addEventListener("click", closeHistoryDrawer);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && historyDrawer && historyDrawer.classList.contains("is-open")) {
        closeHistoryDrawer();
      }
    });

    pipelineButton.addEventListener("click", async function () {
      messageInput.setValue("Show latest pipeline status");
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

    function setSuggestionsVisible(visible) {
      if (!suggestionsElement) return;
      suggestionsElement.hidden = !visible;
      if (promptTriggerElement) {
        promptTriggerElement.hidden = visible;
        promptTriggerElement.setAttribute("aria-expanded", visible ? "true" : "false");
      }
    }

    document.querySelectorAll("[data-chat-prompt]").forEach(function (button) {
      button.addEventListener("click", async function () {
        const prompt = (button.getAttribute("data-chat-prompt") || "").trim();
        if (!prompt) {
          return;
        }
        // Bug #9: populate textarea so the user sees what will be sent,
        // then dispatch the send (matches chatbot_widget chip behaviour).
        messageInput.setValue(prompt);
        await sendMessageFlow(prompt);
      });
    });

    if (suggestionsCloseElement) {
      suggestionsCloseElement.addEventListener("click", function () {
        setSuggestionsVisible(false);
      });
    }

    if (promptTriggerElement) {
      promptTriggerElement.addEventListener("click", function () {
        setSuggestionsVisible(true);
      });
    }

    // Initial state: suggestions visible inline, trigger hidden.
    setSuggestionsVisible(true);

    sessionListElement.addEventListener("scroll", async function () {
      if (state.loadingHistory || !state.hasMoreSessions || state.sessionQuery) {
        return;
      }
      
      const distanceToBottom = sessionListElement.scrollHeight - sessionListElement.scrollTop - sessionListElement.clientHeight;
      if (distanceToBottom <= 20) {
        state.loadingHistory = true;
        
        // Add a visual loading spinner at the bottom
        const spinnerRow = document.createElement("div");
        spinnerRow.id = "session-list-spinner";
        spinnerRow.style.padding = "15px";
        spinnerRow.style.display = "flex";
        spinnerRow.style.justifyContent = "center";
        spinnerRow.innerHTML = '<span class="task-tracker-spinner" aria-hidden="true" style="width:16px;height:16px;opacity:0.7;"></span>';
        sessionListElement.appendChild(spinnerRow);
        
        try {
          await refreshSessionList(true);
        } catch (error) {
          // ignore scroll errors
          if (spinnerRow.parentNode) {
            spinnerRow.remove();
          }
        } finally {
          state.loadingHistory = false;
        }
      }
    });

    renderProjects();
    updateContext();
    setStatus("Loading conversations");
    refreshSessionList()
      .then(async function () {
        // Restore last-used session if it still exists on the server.
        const lastId = loadStoredLastSessionId();
        if (lastId && state.sessions.some(function (s) { return s.session_id === lastId; })) {
          try { await openSession(lastId); } catch (_) { saveStoredLastSessionId(""); }
        } else {
          if (lastId) saveStoredLastSessionId("");
          setStatus("Online · Ready to assist");
        }
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
