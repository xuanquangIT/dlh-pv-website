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

    const dragState = {
      active: false,
      pointerId: null,
      offsetX: 0,
      offsetY: 0,
      moved: false,
      justDragged: false
    };

    function getPortalBounds() {
      var sidebar = document.getElementById("sidebar");
      var topbar = document.getElementById("topbar");
      var minLeft = sidebar ? Math.max(0, sidebar.getBoundingClientRect().right) : 0;
      var minTop = topbar ? Math.max(0, topbar.getBoundingClientRect().bottom) : 0;
      return {
        minLeft: minLeft,
        minTop: minTop
      };
    }

    function applyButtonPosition(left, top) {
      const bounds = getPortalBounds();
      const maxLeft = Math.max(bounds.minLeft, window.innerWidth - openBtn.offsetWidth);
      const maxTop = Math.max(bounds.minTop, window.innerHeight - openBtn.offsetHeight);
      const finalLeft = Math.max(bounds.minLeft, Math.min(maxLeft, left));
      const finalTop = Math.max(bounds.minTop, Math.min(maxTop, top));
      openBtn.style.left = String(finalLeft) + "px";
      openBtn.style.top = String(finalTop) + "px";
      openBtn.style.right = "auto";
      openBtn.style.bottom = "auto";
    }

    function positionPanelNearButton() {
      const bounds = getPortalBounds();
      const btnRect = openBtn.getBoundingClientRect();
      const panelWidth = panel.offsetWidth || 470;
      const panelHeight = panel.offsetHeight || 520;
      const gap = 12;

      let left = btnRect.right - panelWidth;
      left = Math.max(bounds.minLeft + 12, Math.min(window.innerWidth - panelWidth - 12, left));

      let top = btnRect.top - panelHeight - gap;
      if (top < bounds.minTop + 12) {
        top = btnRect.bottom + gap;
      }
      top = Math.max(bounds.minTop + 12, Math.min(window.innerHeight - panelHeight - 12, top));

      panel.style.left = String(left) + "px";
      panel.style.top = String(top) + "px";
      panel.style.right = "auto";
      panel.style.bottom = "auto";
    }

    function bindDrag() {
      openBtn.style.touchAction = "none";

      openBtn.addEventListener("pointerdown", function (event) {
        dragState.active = true;
        dragState.pointerId = event.pointerId;
        dragState.moved = false;

        const rect = openBtn.getBoundingClientRect();
        dragState.offsetX = event.clientX - rect.left;
        dragState.offsetY = event.clientY - rect.top;

        if (openBtn.setPointerCapture) {
          openBtn.setPointerCapture(event.pointerId);
        }
      });

      openBtn.addEventListener("pointermove", function (event) {
        if (!dragState.active || dragState.pointerId !== event.pointerId) {
          return;
        }

        const nextLeft = event.clientX - dragState.offsetX;
        const nextTop = event.clientY - dragState.offsetY;
        applyButtonPosition(nextLeft, nextTop);
        positionPanelNearButton();
        dragState.moved = true;
      });

      function endDrag(event) {
        if (!dragState.active || dragState.pointerId !== event.pointerId) {
          return;
        }

        dragState.active = false;
        if (dragState.moved) {
          dragState.justDragged = true;
          window.setTimeout(function () {
            dragState.justDragged = false;
          }, 120);
        }

        if (openBtn.releasePointerCapture) {
          openBtn.releasePointerCapture(event.pointerId);
        }
        dragState.pointerId = null;
      }

      openBtn.addEventListener("pointerup", endDrag);
      openBtn.addEventListener("pointercancel", endDrag);
    }

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
      if (dragState.justDragged) {
        return;
      }
      panel.classList.add("open");
      panel.setAttribute("aria-hidden", "false");
      positionPanelNearButton();
    });

    closeBtn.addEventListener("click", function () {
      panel.classList.remove("open");
      panel.setAttribute("aria-hidden", "true");
    });

    window.addEventListener("resize", function () {
      positionPanelNearButton();
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
    bindDrag();
    positionPanelNearButton();
  }

  window.PVChatbotWidget = {
    initChatbot: initChatbot
  };
})();
