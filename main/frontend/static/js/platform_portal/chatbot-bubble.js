document.addEventListener("DOMContentLoaded", function () {

var toggle = document.getElementById("chatbot-bubble-toggle");
var panel = document.getElementById("chatbot-bubble-panel");
var closeBtn = document.getElementById("cb-close");
var newSessionBtn = document.getElementById("cb-new-session");
var toggleSessionsBtn = document.getElementById("cb-toggle-sessions");
var sessionDrawer = document.getElementById("cb-session-drawer");
var sessionList = document.getElementById("cb-session-list");
var selectAllSessionsEl = document.getElementById("cb-select-all-sessions");
var deleteSelectedBtn = document.getElementById("cb-delete-selected");
var messagesEl = document.getElementById("cb-messages");
var form = document.getElementById("cb-form");
var roleSelect = document.getElementById("cb-role");
var inputEl = document.getElementById("cb-input");
var sendBtn = document.getElementById("cb-send");

if (!toggle || !panel || !form) return;

var activeSessionId = null;
var isSending = false;
var selectedSessionIds = new Set();

function escapeHtml(text) {
    var d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
}

function logClientError(context, error) {
    var message = error && error.message ? error.message : String(error);
    console.error("[chatbot-bubble] " + context + " failed", message, error);
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
        if (!resp.ok) return;
        var sessions = await resp.json();
        sessionList.innerHTML = "";

        var sessionIdSet = new Set();
        sessions.forEach(function (s) {
            sessionIdSet.add(s.session_id);
            var li = document.createElement("li");
            li.className = "cb-session-item" + (s.session_id === activeSessionId ? " cb-active" : "");

            var checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.className = "cb-session-item-checkbox";
            checkbox.setAttribute("data-session-id", s.session_id);
            checkbox.checked = selectedSessionIds.has(s.session_id);
            checkbox.setAttribute("aria-label", "Select session " + s.title);
            checkbox.addEventListener("click", function (event) {
                event.stopPropagation();
            });
            checkbox.addEventListener("change", function () {
                if (checkbox.checked) {
                    selectedSessionIds.add(s.session_id);
                } else {
                    selectedSessionIds.delete(s.session_id);
                }
                updateSessionSelectionControls(sessionIdSet);
            });

            var title = document.createElement("span");
            title.className = "cb-session-item-title";
            title.textContent = s.title + " (" + s.message_count + ")";

            li.appendChild(checkbox);
            li.appendChild(title);
            li.addEventListener("click", function () {
                selectSession(s.session_id);
                sessionDrawer.classList.add("cb-hidden");
            });
            sessionList.appendChild(li);
        });

        selectedSessionIds.forEach(function (id) {
            if (!sessionIdSet.has(id)) {
                selectedSessionIds.delete(id);
            }
        });

        updateSessionSelectionControls(sessionIdSet);
    } catch (e) {
        logClientError("loadSessions", e);
    }
}

function updateSessionSelectionControls(sessionIdSet) {
    var totalSessions = sessionIdSet ? sessionIdSet.size : 0;
    var selectedCount = selectedSessionIds.size;

    if (deleteSelectedBtn) {
        deleteSelectedBtn.disabled = selectedCount === 0;
        deleteSelectedBtn.textContent = selectedCount > 0 ? "Delete selected (" + selectedCount + ")" : "Delete selected";
    }

    if (selectAllSessionsEl) {
        selectAllSessionsEl.checked = totalSessions > 0 && selectedCount === totalSessions;
        selectAllSessionsEl.indeterminate = selectedCount > 0 && selectedCount < totalSessions;
    }
}

async function deleteSelectedSessions() {
    if (selectedSessionIds.size === 0) return;
    if (!window.confirm("Delete selected chat history items?")) return;

    var idsToDelete = Array.from(selectedSessionIds);
    var activeDeleted = idsToDelete.indexOf(activeSessionId) >= 0;
    deleteSelectedBtn.disabled = true;

    try {
        var results = await Promise.allSettled(idsToDelete.map(function (sessionId) {
            return fetch("/solar-ai-chat/sessions/" + encodeURIComponent(sessionId), {
                method: "DELETE",
            });
        }));

        var hasFailure = results.some(function (result) {
            return result.status !== "fulfilled" || (result.value && !result.value.ok && result.value.status !== 404);
        });

        selectedSessionIds.clear();

        if (activeDeleted) {
            activeSessionId = null;
            renderMessages([]);
        }

        loadSessions();
        if (hasFailure) {
            appendError("Some selected sessions could not be deleted.");
        }
    } catch (error) {
        appendError("Failed to delete selected sessions.");
    }
}

async function selectSession(sessionId) {
    activeSessionId = sessionId;
    try {
        var resp = await fetch("/solar-ai-chat/sessions/" + sessionId);
        if (!resp.ok) return;
        var session = await resp.json();
        renderMessages(session.messages);
    } catch (e) {
        logClientError("selectSession", e);
        appendError("Unable to load the selected session.");
    }
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
    createSession().catch(function (error) {
        logClientError("createSession", error);
        appendError("Failed to start a new chat session.");
    });
});

if (selectAllSessionsEl) {
    selectAllSessionsEl.addEventListener("change", function () {
        var checkboxes = sessionList.querySelectorAll(".cb-session-item-checkbox");
        checkboxes.forEach(function (checkbox) {
            checkbox.checked = selectAllSessionsEl.checked;
            var sessionId = checkbox.getAttribute("data-session-id");
            if (sessionId) {
                if (selectAllSessionsEl.checked) {
                    selectedSessionIds.add(sessionId);
                } else {
                    selectedSessionIds.delete(sessionId);
                }
            }
        });
        updateSessionSelectionControls(new Set(Array.from(checkboxes).map(function (cb) {
            return cb.getAttribute("data-session-id");
        }).filter(Boolean)));
    });
}

if (deleteSelectedBtn) {
    deleteSelectedBtn.addEventListener("click", function () {
        deleteSelectedSessions().catch(function (error) {
            logClientError("deleteSelectedSessions", error);
            appendError("Failed to delete selected sessions.");
        });
    });
}

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
