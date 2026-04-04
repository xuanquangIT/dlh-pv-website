(function () {
  function initChatbot() {
    const shared = window.PVSolarChatPage;
    if (!shared) {
      return;
    }

    const getActiveRole = shared.getActiveRole;
    const createWelcomeMessage = shared.createWelcomeMessage;
    const createMessageElement = shared.createMessageElement;
    const SolarChatApi = shared.SolarChatApi;
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
      role: getActiveRole(),
      sessionId: "",
      loading: false,
      messages: [createWelcomeMessage()]
    };

    function syncWidgetRoleFromGlobal() {
      const activeRole = getActiveRole();
      if (activeRole !== widgetState.role) {
        widgetState.role = activeRole;
        widgetState.sessionId = "";
      }
    }

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

      syncWidgetRoleFromGlobal();

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

    window.addEventListener("pv-role-changed", function (event) {
      const role = event && event.detail ? event.detail.role : "";
      if (role && role !== widgetState.role) {
        widgetState.role = role;
        widgetState.sessionId = "";
      }
    });

    normalizeInitialMessages();
  }

  window.PVChatbotWidget = {
    initChatbot: initChatbot
  };
})();
