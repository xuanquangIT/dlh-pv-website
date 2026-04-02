document.addEventListener("DOMContentLoaded", function () {

const form = document.getElementById("chat-form");
const roleSelect = document.getElementById("role");
const messageArea = document.getElementById("message");
const statusText = document.getElementById("status-text");
const responseOutput = document.getElementById("response-output");
const submitButton = document.getElementById("submit-button");
const sessionList = document.getElementById("session-list");
const newSessionBtn = document.getElementById("new-session-btn");
const forkSessionBtn = document.getElementById("fork-session-btn");
const chatMessages = document.getElementById("chat-messages");
const sessionTitleLabel = document.getElementById("session-title-label");

if (!form || !roleSelect || !messageArea || !submitButton) {
    console.error("Solar AI Chat: required DOM elements not found. Check element IDs.");
    return;
}

let activeSessionId = null;

function setPendingState(isPending) {
    submitButton.disabled = isPending;
    submitButton.textContent = isPending ? "Sending..." : "Send Query";
}

function setStatus(message, isError) {
    statusText.textContent = message;
    statusText.className = "status-text" + (isError ? " status-error" : " status-ok");
}

function appendErrorBubble(message) {
    const existing = chatMessages.querySelector(".empty-state");
    if (existing) existing.remove();
    const div = document.createElement("div");
    div.className = "chat-msg chat-msg-error";
    div.innerHTML = "<strong>Error:</strong> " + escapeHtml(message);
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function renderMessage(msg) {
    const div = document.createElement("div");
    div.className = "chat-msg chat-msg-" + msg.sender;
    const label = msg.sender === "user" ? "You" : "Assistant";
    let html = "<strong>" + escapeHtml(label) + ":</strong> " + escapeHtml(msg.content);
    if (msg.sender === "assistant" && Array.isArray(msg.sources) && msg.sources.length > 0) {
        const uniqueSources = msg.sources.filter(
            (s, i, arr) => arr.findIndex(x => x.data_source === s.data_source) === i
        );
        const tags = uniqueSources.map(function (s) {
            const cls = s.data_source === "trino" ? "source-tag source-tag-trino" : "source-tag source-tag-csv";
            return '<span class="' + cls + '">' + escapeHtml(s.data_source.toUpperCase()) + '</span>';
        }).join(" ");
        html += ' <span class="source-tag-row">' + tags + '</span>';
    }
    div.innerHTML = html;
    return div;
}

function renderMessages(messages) {
    chatMessages.innerHTML = "";
    if (!messages || messages.length === 0) {
        chatMessages.innerHTML = '<p class="empty-state">No messages yet. Send a question to start.</p>';
        return;
    }
    messages.forEach(function (msg) {
        chatMessages.appendChild(renderMessage(msg));
    });
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function loadSessions() {
    try {
        const response = await fetch("/solar-ai-chat/sessions");
        const sessions = await response.json();
        sessionList.innerHTML = "";
        sessions.forEach(function (session) {
            const li = document.createElement("li");
            li.className = "session-item" + (session.session_id === activeSessionId ? " active" : "");
            li.dataset.sessionId = session.session_id;

            const titleSpan = document.createElement("span");
            titleSpan.className = "session-title";
            titleSpan.textContent = session.title + " (" + session.message_count + ")";
            li.appendChild(titleSpan);

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "btn-delete";
            deleteBtn.textContent = "X";
            deleteBtn.title = "Delete session";
            deleteBtn.addEventListener("click", function (e) {
                e.stopPropagation();
                deleteSession(session.session_id);
            });
            li.appendChild(deleteBtn);

            li.addEventListener("click", function () {
                selectSession(session.session_id);
            });
            sessionList.appendChild(li);
        });
    } catch (err) {
        statusText.textContent = "Failed to load sessions.";
    }
}

async function selectSession(sessionId) {
    activeSessionId = sessionId;
    forkSessionBtn.disabled = false;
    try {
        const response = await fetch("/solar-ai-chat/sessions/" + sessionId);
        if (!response.ok) {
            statusText.textContent = "Failed to load session.";
            return;
        }
        const session = await response.json();
        sessionTitleLabel.textContent = "- " + session.title;
        renderMessages(session.messages);
        loadSessions();
    } catch (err) {
        statusText.textContent = "Failed to load session.";
    }
}

async function createSession() {
    const role = roleSelect.value.trim();
    const title = "Chat " + new Date().toLocaleString();
    const response = await fetch("/solar-ai-chat/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: role, title: title }),
    });
    if (!response.ok) {
        const body = await response.json().catch(function () { return {}; });
        throw new Error(body.detail || "Failed to create session.");
    }
    const session = await response.json();
    activeSessionId = session.session_id;
    forkSessionBtn.disabled = false;
    sessionTitleLabel.textContent = "- " + session.title;
    renderMessages([]);
    await loadSessions();
}

async function deleteSession(sessionId) {
    try {
        await fetch("/solar-ai-chat/sessions/" + sessionId, { method: "DELETE" });
        if (activeSessionId === sessionId) {
            activeSessionId = null;
            forkSessionBtn.disabled = true;
            sessionTitleLabel.textContent = "";
            chatMessages.innerHTML = '<p class="empty-state">Select or create a session to start chatting.</p>';
        }
        await loadSessions();
    } catch (err) {
        statusText.textContent = "Failed to delete session.";
    }
}

async function forkSession() {
    if (!activeSessionId) return;
    const role = roleSelect.value.trim();
    const title = "Fork " + new Date().toLocaleString();
    try {
        const response = await fetch("/solar-ai-chat/sessions/" + activeSessionId + "/fork", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ role: role, title: title }),
        });
        const body = await response.json().catch(function () { return {}; });
        if (!response.ok) {
            const detail = typeof body.detail === "string" ? body.detail : "Failed to fork session.";
            throw new Error(detail);
        }
        await selectSession(body.session_id);
        await loadSessions();
        setStatus("Session forked successfully.", false);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setStatus(msg, true);
        appendErrorBubble(msg);
    }
}

async function querySolarChat(payload) {
    const response = await fetch("/solar-ai-chat/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
        const detail = typeof body.detail === "string" ? body.detail : "Request failed.";
        throw new Error(detail);
    }
    return body;
}

form.addEventListener("submit", async function (event) {
    event.preventDefault();
    setPendingState(true);
    try {
        const role = roleSelect.value.trim();
        const message = messageArea.value.trim();
        if (!role || !message) {
            setStatus("Role and message are required.", true);
            return;
        }

        if (!activeSessionId) {
            await createSession();
        }

        const payload = {
            role: role,
            session_id: activeSessionId,
            message: message,
        };

        setStatus("Sending request...", false);
        responseOutput.textContent = "Waiting for response...";

        const result = await querySolarChat(payload);
        const sourceLabel = Array.isArray(result.sources) && result.sources.length > 0
            ? result.sources[0].data_source.toUpperCase()
            : "unknown";
        setStatus("OK  Topic: " + result.topic + "  |  Model: " + result.model_used + "  |  Source: " + sourceLabel, false);
        responseOutput.textContent = JSON.stringify(result, null, 2);
        messageArea.value = "";
        await selectSession(activeSessionId);
    } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        setStatus("Request failed.", true);
        appendErrorBubble(msg);
        responseOutput.textContent = JSON.stringify({ error: msg }, null, 2);
    } finally {
        setPendingState(false);
    }
});

newSessionBtn.addEventListener("click", function () {
    createSession().catch(function (err) {
        setStatus(err.message || "Failed to create session.", true);
        appendErrorBubble(err.message || "Failed to create session.");
    });
});
forkSessionBtn.addEventListener("click", forkSession);

loadSessions();

}); // end DOMContentLoaded
