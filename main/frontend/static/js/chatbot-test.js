const form = document.getElementById("chat-form");
const statusText = document.getElementById("status-text");
const responseOutput = document.getElementById("response-output");
const submitButton = document.getElementById("submit-button");

function setPendingState(isPending) {
    submitButton.disabled = isPending;
    submitButton.textContent = isPending ? "Sending..." : "Send Query";
}

async function querySolarChat(payload) {
    const response = await fetch("/solar-ai-chat/query", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    const body = await response.json();

    if (!response.ok) {
        const detail = typeof body.detail === "string" ? body.detail : "Request failed.";
        throw new Error(detail);
    }

    return body;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const role = form.role.value.trim();
    const sessionId = form.session_id.value.trim();
    const message = form.message.value.trim();

    if (!role || !message) {
        statusText.textContent = "Role and message are required.";
        return;
    }

    const payload = {
        role,
        session_id: sessionId || "manual-ui-session",
        message,
    };

    statusText.textContent = "Sending request...";
    responseOutput.textContent = "Waiting for response...";
    setPendingState(true);

    try {
        const result = await querySolarChat(payload);
        statusText.textContent = `Success. Topic: ${result.topic} | Model: ${result.model_used}`;
        responseOutput.textContent = JSON.stringify(result, null, 2);
    } catch (error) {
        statusText.textContent = "Request failed.";
        responseOutput.textContent = JSON.stringify(
            { error: error instanceof Error ? error.message : String(error) },
            null,
            2,
        );
    } finally {
        setPendingState(false);
    }
});
