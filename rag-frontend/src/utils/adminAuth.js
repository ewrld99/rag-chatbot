import { API_BASE, BACKEND_UNAVAILABLE_MESSAGE } from "../api/chatApi";

const ADMIN_SESSION_KEY = "ragAdminSession";

function base64UrlDecode(value) {
    const normalized = value.replaceAll("-", "+").replaceAll("_", "/");
    const padded = normalized.padEnd(normalized.length + ((4 - (normalized.length % 4)) % 4), "=");
    return JSON.parse(window.atob(padded));
}

export function getAdminSession() {
    try {
        const raw = window.localStorage.getItem(ADMIN_SESSION_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch {
        window.localStorage.removeItem(ADMIN_SESSION_KEY);
        return null;
    }
}

export function getAdminToken() {
    return getAdminSession()?.token || "";
}

export function saveAdminSession(user) {
    if (!user?.token) {
        throw new Error("Admin login did not return a session token.");
    }
    const session = {
        id: user.id,
        username: user.username,
        role: "admin",
        token: user.token,
    };

    window.localStorage.setItem(ADMIN_SESSION_KEY, JSON.stringify(session));
    return session;
}

export function isAdminAuthenticated() {
    const session = getAdminSession();

    if (!session?.token) {
        return false;
    }

    try {
        const [, payload] = session.token.split(".");
        const claims = base64UrlDecode(payload);
        const isValid = claims.role === "admin" && claims.exp > Math.floor(Date.now() / 1000);

        if (!isValid) {
            clearAdminSession();
        }

        return isValid;
    } catch {
        clearAdminSession();
        return false;
    }
}

function formatAuthError(data) {
    if (typeof data?.detail === "string") {
        return data.detail;
    }

    if (Array.isArray(data?.detail)) {
        return data.detail
            .map((item) => item?.msg)
            .filter(Boolean)
            .join(" ") || "Invalid admin credentials.";
    }

    return "Invalid admin credentials.";
}

export async function loginAdmin(username, password) {
    let res;

    try {
        res = await fetch(`${API_BASE}/api/auth/login`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ username, password }),
        });
    } catch {
        throw new Error(BACKEND_UNAVAILABLE_MESSAGE);
    }

    const data = await res.json();

    if (!res.ok) {
        throw new Error(formatAuthError(data));
    }

    if (data.role !== "admin") {
        throw new Error("This account does not have admin access.");
    }

    return saveAdminSession(data);
}

export function clearAdminSession() {
    window.localStorage.removeItem(ADMIN_SESSION_KEY);
}
