(function () {
    function getSessionUser() {
        try {
            var raw = window.sessionStorage.getItem("pv_user");
            if (!raw) {
                return null;
            }
            var parsed = JSON.parse(raw);
            return parsed && typeof parsed === "object" ? parsed : null;
        } catch (error) {
            return null;
        }
    }

    function getNextPath() {
        var params = new URLSearchParams(window.location.search);
        var next = params.get("next") || "/dashboard";
        if (!next.startsWith("/")) {
            return "/dashboard";
        }
        return next;
    }

    function byId(id) {
        return document.getElementById(id);
    }

    function setVisible(element, visible, message) {
        if (!element) {
            return;
        }
        if (message) {
            element.textContent = message;
        }
        element.classList.toggle("visible", visible);
    }

    function parseDisplayNameFromEmail(email) {
        var handle = (email || "").split("@")[0].trim();
        if (!handle) {
            return "Platform User";
        }
        return handle
            .split(/[._-]+/)
            .filter(Boolean)
            .map(function (part) {
                return part.charAt(0).toUpperCase() + part.slice(1).toLowerCase();
            })
            .join(" ");
    }

    function makeInitials(name) {
        var chunks = String(name || "")
            .split(/\s+/)
            .filter(Boolean);
        if (!chunks.length) {
            return "PU";
        }
        return chunks
            .slice(0, 2)
            .map(function (part) {
                return part.charAt(0).toUpperCase();
            })
            .join("");
    }

    function saveSession(user, remember) {
        var payload = {
            name: user.name,
            email: user.email,
            role: user.role,
            initials: user.initials,
            avatarClass: user.avatarClass,
            signedAt: new Date().toISOString(),
            remember: remember,
        };
        window.sessionStorage.setItem("pv_user", JSON.stringify(payload));
        if (remember) {
            window.localStorage.setItem("pv_user_hint", user.email);
        } else {
            window.localStorage.removeItem("pv_user_hint");
        }
    }

    function redirectToDashboard() {
        window.location.assign(getNextPath());
    }

    function setLoading(button, loading) {
        if (!button) {
            return;
        }
        button.classList.toggle("loading", loading);
        button.disabled = loading;
        button.textContent = loading ? "Signing In..." : "Sign In";
    }

    function runPasswordToggle() {
        var toggleButton = byId("toggle-password");
        var passwordInput = byId("password");

        if (!toggleButton || !passwordInput) {
            return;
        }

        toggleButton.addEventListener("click", function () {
            var reveal = passwordInput.type === "password";
            passwordInput.type = reveal ? "text" : "password";
            toggleButton.textContent = reveal ? "Hide" : "Show";
            toggleButton.setAttribute("aria-label", reveal ? "Hide password" : "Show password");
        });
    }

    function runLogoutPage() {
        var confirmButton = byId("confirm-logout-btn");
        if (!confirmButton) {
            return;
        }

        var sessionUser = getSessionUser();
        if (sessionUser) {
            var initials = byId("logout-user-initials");
            var name = byId("logout-user-name");
            var role = byId("logout-user-role");

            if (initials && sessionUser.initials) {
                initials.textContent = sessionUser.initials;
            }
            if (name && sessionUser.name) {
                name.textContent = sessionUser.name;
            }
            if (role && sessionUser.role) {
                role.textContent = sessionUser.role;
            }
        }

        confirmButton.addEventListener("click", function () {
            window.sessionStorage.removeItem("pv_user");
            window.location.assign("/login?logged_out=1");
        });
    }

    function runLoginPage() {
        var form = byId("login-form");
        if (!form) {
            return;
        }

        var emailInput = byId("email");
        var passwordInput = byId("password");
        var rememberInput = byId("remember");
        var signInButton = byId("sign-in-btn");
        var errorBox = byId("login-error");
        var flashBox = byId("login-flash");

        var params = new URLSearchParams(window.location.search);
        if (params.get("logged_out") === "1") {
            setVisible(flashBox, true, "You have successfully signed out.");
        }

        if (getSessionUser()) {
            redirectToDashboard();
            return;
        }

        var storedHint = window.localStorage.getItem("pv_user_hint");
        if (storedHint && emailInput) {
            emailInput.value = storedHint;
            if (rememberInput) {
                rememberInput.checked = true;
            }
        }

        form.addEventListener("submit", function (event) {
            event.preventDefault();
            setVisible(errorBox, false, "");

            var email = emailInput ? emailInput.value.trim() : "";
            var password = passwordInput ? passwordInput.value : "";

            if (!email || !password) {
                setVisible(errorBox, true, "Please enter your email and password.");
                return;
            }

            if (!email.includes("@") || password.length < 4) {
                setVisible(errorBox, true, "Invalid email or password. Please try again.");
                return;
            }

            setLoading(signInButton, true);

            var name = parseDisplayNameFromEmail(email);
            var user = {
                name: name,
                email: email,
                role: "Engineer",
                initials: makeInitials(name),
                avatarClass: "avatar-blue",
            };

            window.setTimeout(function () {
                saveSession(user, Boolean(rememberInput && rememberInput.checked));
                redirectToDashboard();
            }, 700);
        });

        document.querySelectorAll(".demo-user-card").forEach(function (card) {
            card.addEventListener("click", function () {
                var user = {
                    name: card.dataset.name || "Platform User",
                    email: card.dataset.email || "user@pvlakehouse.io",
                    role: card.dataset.role || "Engineer",
                    initials: card.dataset.initials || "PU",
                    avatarClass: card.dataset.avatarClass || "avatar-blue",
                };
                saveSession(user, false);
                redirectToDashboard();
            });
        });

        runPasswordToggle();
    }

    document.addEventListener("DOMContentLoaded", function () {
        runLoginPage();
        runLogoutPage();
    });
})();
