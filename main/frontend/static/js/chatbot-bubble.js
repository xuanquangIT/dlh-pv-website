document.addEventListener("DOMContentLoaded", function () {

var toggle = document.getElementById("chatbot-bubble-toggle");
var panel = document.getElementById("chatbot-bubble-panel");
var closeBtn = document.getElementById("cb-close");
var newSessionBtn = document.getElementById("cb-new-session");
var toggleSessionsBtn = document.getElementById("cb-toggle-sessions");
var sessionDrawer = document.getElementById("cb-session-drawer");
var sessionList = document.getElementById("cb-session-list");
var messagesEl = document.getElementById("cb-messages");
var form = document.getElementById("cb-form");
var roleSelect = document.getElementById("cb-role");
var inputEl = document.getElementById("cb-input");
var sendBtn = document.getElementById("cb-send");

if (!toggle || !panel || !form) return;

var activeSessionId = null;
var isSending = false;

function escapeHtml(text) {
    var d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
}

// Panel open/close
toggle.addEventListener("click", function () {
    panel.classList.toggle("cb-hidden");
    toggle.classList.toggle("cb-toggle-active");
    if (!panel.classList.contains("cb-hidden")) {
        inputEl.focus();
        loadSessions();
    }
});

closeBtn.addEventListener("click", function () {
    panel.classList.add("cb-hidden");
    toggle.classList.remove("cb-toggle-active");
});

// Session drawer toggle
toggleSessionsBtn.addEventListener("click", function () {
    sessionDrawer.classList.toggle("cb-hidden");
    if (!sessionDrawer.classList.contains("cb-hidden")) {
        loadSessions();
    }
});

// Sessions API
async function loadSessions() {
    try {
        var resp = await fetch("/solar-ai-chat/sessions");
        var sessions = await resp.json();
        sessionList.innerHTML = "";
        sessions.forEach(function (s) {
            var li = document.createElement("li");
            li.className = "cb-session-item" + (s.session_id === activeSessionId ? " cb-active" : "");
            li.textContent = s.title + " (" + s.message_count + ")";
            li.addEventListener("click", function () {
                selectSession(s.session_id);
                sessionDrawer.classList.add("cb-hidden");
            });
            sessionList.appendChild(li);
        });
    } catch (e) { /* silent */ }
}

async function selectSession(sessionId) {
    activeSessionId = sessionId;
    try {
        var resp = await fetch("/solar-ai-chat/sessions/" + sessionId);
        if (!resp.ok) return;
        var session = await resp.json();
        renderMessages(session.messages);
    } catch (e) { /* silent */ }
}

async function createSession() {
    var role = roleSelect.value;
    var title = "Chat " + new Date().toLocaleTimeString();
    var resp = await fetch("/solar-ai-chat/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: role, title: title }),
    });
    if (!resp.ok) throw new Error("Failed to create session");
    var session = await resp.json();
    activeSessionId = session.session_id;
    renderMessages([]);
    loadSessions();
}

newSessionBtn.addEventListener("click", function () {
    createSession().catch(function () {});
});

// Message rendering
function renderMessages(msgs) {
    messagesEl.innerHTML = "";
    if (!msgs || msgs.length === 0) {
        messagesEl.innerHTML = '<p class="cb-empty">Ask a question to start chatting.</p>';
        return;
    }
    msgs.forEach(function (msg) {
        appendMessage(msg.sender, msg.content, msg.sources);
    });
}

function appendMessage(sender, content, sources) {
    var existing = messagesEl.querySelector(".cb-empty");
    if (existing) existing.remove();

    var div = document.createElement("div");
    div.className = "cb-msg cb-msg-" + sender;

    var html = escapeHtml(content);
    if (sender === "assistant" && Array.isArray(sources) && sources.length > 0) {
        var ds = sources[0].data_source || "unknown";
        var cls = ds === "trino" ? "source-tag source-tag-trino" : "source-tag source-tag-csv";
        html += ' <span class="' + cls + '">' + escapeHtml(ds.toUpperCase()) + '</span>';
    }
    div.innerHTML = html;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendError(message) {
    var existing = messagesEl.querySelector(".cb-empty");
    if (existing) existing.remove();
    var div = document.createElement("div");
    div.className = "cb-msg cb-msg-error";
    div.textContent = message;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// Send query
form.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (isSending) return;

    var message = inputEl.value.trim();
    if (!message) return;

    isSending = true;
    sendBtn.disabled = true;

    try {
        if (!activeSessionId) {
            await createSession();
        }

        appendMessage("user", message);
        inputEl.value = "";

        // Typing indicator
        var typing = document.createElement("div");
        typing.className = "cb-msg cb-msg-assistant cb-typing";
        typing.textContent = "Thinking...";
        messagesEl.appendChild(typing);
        messagesEl.scrollTop = messagesEl.scrollHeight;

        var resp = await fetch("/solar-ai-chat/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                role: roleSelect.value,
                session_id: activeSessionId,
                message: message,
            }),
        });

        // Remove typing indicator
        if (typing.parentNode) typing.remove();

        var body = await resp.json();
        if (!resp.ok) {
            var detail = typeof body.detail === "string" ? body.detail : "Request failed.";
            appendError(detail);
            return;
        }

        appendMessage("assistant", body.answer, body.sources);
    } catch (err) {
        // Remove typing indicator if still present
        var typingEl = messagesEl.querySelector(".cb-typing");
        if (typingEl) typingEl.remove();
        appendError(err.message || "Connection error.");
    } finally {
        isSending = false;
        sendBtn.disabled = false;
        inputEl.focus();
    }
});

// Enter to send
inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.dispatchEvent(new Event("submit"));
    }
});

}); // end DOMContentLoaded
