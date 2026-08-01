function localBackendBase() {
    if (typeof window === "undefined") return "http://127.0.0.1:8000";
    const { hostname, origin, protocol, port } = window.location;
    const localBackendHost = hostname === "localhost" ? "127.0.0.1" : hostname;
    if (port === "5173" || port === "4173") {
        return `${protocol}//${localBackendHost}:8000`;
    }
    if (hostname === "localhost" || hostname === "127.0.0.1") {
        return `${protocol}//${localBackendHost}:8000`;
    }
    return origin;
}

function trimTrailingSlash(value) {
    return value.replace(/\/+$/, "");
}

const configuredApiBase = import.meta.env.VITE_API_BASE?.trim();
export const API_BASE = trimTrailingSlash(configuredApiBase || localBackendBase());

function websocketBase() {
    if (import.meta.env.VITE_WS_BASE?.trim()) {
        return import.meta.env.VITE_WS_BASE.trim();
    }
    const origin = typeof window === "undefined" ? API_BASE : window.location.origin;
    const url = new URL(API_BASE, origin);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = "/ws/chat";
    url.search = "";
    url.hash = "";
    return url.toString();
}

export const WS_BASE = trimTrailingSlash(websocketBase());
export const BACKEND_UNAVAILABLE_MESSAGE = `Cannot reach the backend server at ${API_BASE}. Make sure FastAPI is running and VITE_API_BASE is correct.`;

function storedTokenFrom(key) {
    if (typeof window === "undefined") return "";
    try {
        const raw = window.localStorage.getItem(key);
        const session = raw ? JSON.parse(raw) : null;
        return session?.token || "";
    } catch {
        return "";
    }
}

export function getAuthToken() {
    return storedTokenFrom("ragUser") || storedTokenFrom("ragAdminSession");
}

export function authHeaders() {
    const token = getAuthToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
}

function formatApiError(data, fallback = "Request failed") {
    const detail = data?.detail;

    if (!detail) {
        return fallback;
    }

    if (typeof detail === "string") {
        return detail;
    }

    if (Array.isArray(detail)) {
        return detail
            .map((item) => {
                const field = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : null;
                return field ? `${field}: ${item.msg}` : item.msg;
            })
            .filter(Boolean)
            .join(". ");
    }

    if (typeof detail === "object") {
        return detail.message || detail.error || JSON.stringify(detail);
    }

    return String(detail);
}

async function request(path, options = {}) {
    let res;

    try {
        res = await fetch(`${API_BASE}${path}`, {
            headers: {
                "Content-Type": "application/json",
                ...authHeaders(),
                ...options.headers,
            },
            ...options,
        });
    } catch {
        throw new Error(BACKEND_UNAVAILABLE_MESSAGE);
    }

    const contentType = res.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
        ? await res.json()
        : { detail: await res.text() };

    if (!res.ok) {
        throw new Error(formatApiError(data, `Request failed with status ${res.status}`));
    }

    return data;
}

export function registerUser(username, password, registrationNumber = null, programme = null, campus = null, admissionYear = null) {
    return request("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({ 
            username, 
            password,
            registration_number: registrationNumber,
            programme: programme,
            campus: campus,
            admission_year: admissionYear ? parseInt(admissionYear, 10) : null
        }),
    });
}

export function loginUser(username, password) {
    return request("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
    });
}

export function changePassword(userId, oldPassword, newPassword) {
    return request("/api/auth/change-password", {
        method: "POST",
        body: JSON.stringify({
            user_id: userId,
            old_password: oldPassword,
            new_password: newPassword,
        }),
    });
}

export function submitMessageFeedback(messageId, feedbackValue) {
    return request(`/api/chat/messages/${messageId}/feedback`, {
        method: "POST",
        body: JSON.stringify({ feedback: feedbackValue }),
    });
}

export function getFeedbackMessages(skip = 0, limit = 100) {
    return request(`/api/chat/feedback/messages?skip=${skip}&limit=${limit}`);
}

export function getChatSessions(userId) {
    return request(`/api/chat/users/${userId}/sessions`);
}

export function createChatSession(userId, title = "New chat") {
    return request(`/api/chat/users/${userId}/sessions`, {
        method: "POST",
        body: JSON.stringify({ title }),
    });
}

export function getChatSession(userId, sessionId) {
    return request(`/api/chat/users/${userId}/sessions/${sessionId}`);
}

export function deleteChatSession(userId, sessionId) {
    return request(`/api/chat/users/${userId}/sessions/${sessionId}`, {
        method: "DELETE",
    });
}
