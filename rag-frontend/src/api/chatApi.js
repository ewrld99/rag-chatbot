export const API_BASE = "http://localhost:8000";
export const WS_BASE = "ws://localhost:8000/ws/chat";

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
                ...options.headers,
            },
            ...options,
        });
    } catch {
        throw new Error("Cannot reach the backend server. Make sure FastAPI is running on http://localhost:8000.");
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
