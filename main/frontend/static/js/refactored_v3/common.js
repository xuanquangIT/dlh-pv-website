(function () {
  const colors = {
    solar: "#f4b942",
    green: "#1a8a5a",
    blue: "#1b6ca8",
    orange: "#e07b39",
    red: "#c0392b"
  };

  const SOLAR_WELCOME_MESSAGE =
    "👋 Xin chào! Tôi là **Solar AI**, trợ lý thông minh của PV Lakehouse.\n\n" +
    "Tôi có thể giúp bạn phân tích dữ liệu năng lượng, kiểm tra pipeline, xem metrics mô hình, hoặc giải thích các dự báo. Bạn cần hỗ trợ gì?";

  const SolarChatApi = {
    async createSession(role, title) {
      return requestJson("/solar-ai-chat/sessions", {
        method: "POST",
        body: JSON.stringify({ role: role, title: title })
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
    }
  };

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
        detailMessage = (errorData && errorData.detail) || detailMessage;
      } catch (e) {
        detailMessage = response.statusText || detailMessage;
      }
      throw new Error(detailMessage);
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
    const clearButton = document.getElementById("solar-chat-clear-btn");
    const exportButton = document.getElementById("solar-chat-export-btn");
    const pipelineButton = document.getElementById("pipeline-status-btn");

    if (
      !messagesElement ||
      !inputElement ||
      !sendButton ||
      !statusElement ||
      !errorElement ||
      !loadingElement ||
      !clearButton ||
      !exportButton ||
      !pipelineButton
    ) {
      return;
    }

    const state = {
      role: "viewer",
      sessionId: "",
      messages: [createWelcomeMessage()],
      loading: false,
      modelUsed: ""
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

    async function ensureSession() {
      if (state.sessionId) {
        return state.sessionId;
      }
      const now = new Date().toISOString().slice(0, 19).replace("T", " ");
      const created = await SolarChatApi.createSession(state.role, "Solar chat " + now);
      state.sessionId = created.session_id;
      updateContext();
      return state.sessionId;
    }

    async function loadSessionMessages(sessionId) {
      const detail = await SolarChatApi.getSession(sessionId);
      const loadedMessages = (detail.messages || []).map(function (message) {
        const mappedRole = message.sender === "assistant" ? "assistant" : "user";
        return {
          role: mappedRole,
          content: message.content || "",
          timestamp: message.timestamp
        };
      });
      state.messages = withWelcomeMessage(loadedMessages);
      messageList.render(state.messages);
      updateContext();
    }

    async function sendMessageFlow(text) {
      if (state.loading) {
        return;
      }

      try {
        setError("");
        setLoading(true, "Sending message to assistant...");
        setStatus("Processing");

        const sessionId = await ensureSession();

        // Optimistic message append while waiting API.
        state.messages.push({ role: "user", content: text });
        messageList.render(state.messages);
        messageInput.clear();
        updateContext();

        const response = await SolarChatApi.query({
          message: text,
          role: state.role,
          session_id: sessionId
        });

        state.modelUsed = response.model_used || state.modelUsed;
        updateContext();

        setLoading(true, "Loading full session history...");
        await loadSessionMessages(sessionId);
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
    }

    clearButton.addEventListener("click", async function () {
      await resetConversation();
    });

    exportButton.addEventListener("click", function () {
      setStatus("Export is not implemented yet.");
    });

    pipelineButton.addEventListener("click", async function () {
      await sendMessageFlow("Show latest pipeline status");
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

    updateContext();
    setStatus("Initializing session");
    ensureSession()
      .then(() => {
        setStatus("Online · Ready to assist");
      })
      .catch((error) => {
        setStatus("Error");
        setError(error instanceof Error ? error.message : "Failed to create chat session.");
      });
  }

  function initChatbot() {
    const panel = document.getElementById("chatbot-panel");
    const openBtn = document.getElementById("chatbot-btn");
    const closeBtn = document.getElementById("chatbot-close");
    const input = document.getElementById("cbot-input");
    const sendBtn = document.getElementById("cbot-send");
    const messages = document.getElementById("cbot-msgs");

    if (!panel || !openBtn || !closeBtn || !input || !sendBtn || !messages) {
      return;
    }

    const widgetState = {
      role: "viewer",
      sessionId: "",
      loading: false,
      messages: [createWelcomeMessage()]
    };

    function setWidgetLoading(loading) {
      widgetState.loading = loading;
      sendBtn.disabled = loading;
      input.disabled = loading;
    }

    function renderWidgetMessages() {
      messages.innerHTML = "";
      widgetState.messages.forEach(function (message) {
        messages.appendChild(createMessageElement(message.role, message.content, message.timestamp));
      });
      messages.scrollTop = messages.scrollHeight;
    }

    function appendMessage(role, content) {
      widgetState.messages.push({
        role: role,
        content: content,
        timestamp: new Date().toISOString()
      });
      renderWidgetMessages();
    }

    function normalizeInitialMessages() {
      widgetState.messages = [createWelcomeMessage()];
      renderWidgetMessages();
    }

    async function ensureWidgetSession() {
      if (widgetState.sessionId) {
        return widgetState.sessionId;
      }
      const created = await SolarChatApi.createSession(widgetState.role, "Floating chat session");
      widgetState.sessionId = created.session_id;
      return widgetState.sessionId;
    }

    async function sendWidgetMessage() {
      if (widgetState.loading) {
        return;
      }

      const text = input.value.trim();
      if (!text) {
        return;
      }

      try {
        setWidgetLoading(true);
        appendMessage("user", text);
        input.value = "";

        const sessionId = await ensureWidgetSession();
        const response = await SolarChatApi.query({
          message: text,
          role: widgetState.role,
          session_id: sessionId
        });
        appendMessage("assistant", response.answer || "No answer available.");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Request failed.";
        appendMessage("assistant", "Request failed: " + message);
      } finally {
        setWidgetLoading(false);
      }
    }

    openBtn.addEventListener("click", function () {
      panel.classList.add("open");
      panel.setAttribute("aria-hidden", "false");
    });

    closeBtn.addEventListener("click", function () {
      panel.classList.remove("open");
      panel.setAttribute("aria-hidden", "true");
    });

    sendBtn.addEventListener("click", sendWidgetMessage);
    input.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendWidgetMessage();
      }
    });

    document.querySelectorAll(".cbot-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        const prompt = chip.getAttribute("data-chat-prompt") || chip.textContent || "";
        input.value = prompt.replace(/\s+/g, " ").trim();
        sendWidgetMessage();
      });
    });

    normalizeInitialMessages();
  }

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

  function initAnalyticsChart() {
    const canvas = document.getElementById("analyticsChart");
    if (!canvas || typeof Chart === "undefined") return;
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: ["Jan 6", "Jan 7", "Jan 8"],
        datasets: [
          { label: "RSA", data: [1312, 1198, 1240], backgroundColor: "rgba(244,185,66,.8)" },
          { label: "MVB", data: [1042, 994, 1018], backgroundColor: "rgba(27,108,168,.8)" },
          { label: "DPC", data: [902, 861, 882], backgroundColor: "rgba(26,138,90,.8)" }
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
        labels: ["RMSE", "MAE", "R2", "MAPE", "Speed", "Memory"],
        datasets: [
          { label: "GBT-v4.2", data: [92, 90, 94, 95, 80, 75], borderColor: colors.orange, backgroundColor: "rgba(224,123,57,.1)" },
          { label: "GBT-v4.1", data: [84, 83, 88, 87, 82, 78], borderColor: colors.blue, backgroundColor: "rgba(27,108,168,.1)" }
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

  document.addEventListener("DOMContentLoaded", function () {
    initChatbot();
    bindTopbarActions();

    switch (window.PV_ROUTE) {
      case "dashboard":
        initDashboardCharts();
        break;
      case "training":
        initTrainingChart();
        break;
      case "forecast":
        initForecastChart();
        break;
      case "analytics":
        initAnalyticsChart();
        break;
      case "registry":
        initCompareChart();
        break;
      case "solar_chat":
        initSolarChatPage();
        break;
      default:
        break;
    }
  });
})();
