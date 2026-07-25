import { API_BASE } from "./chatApi";
import { getAdminToken } from "../utils/adminAuth";

async function requestAlias(path, options = {}) {
    let res;
    const headers = {
        ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...(getAdminToken() ? { Authorization: `Bearer ${getAdminToken()}` } : {}),
        ...options.headers,
    };

    try {
        res = await fetch(`${API_BASE}/api/admin/retrieval-aliases${path}`, {
            ...options,
            headers,
        });
    } catch {
        throw new Error("Cannot reach the backend server. Make sure FastAPI is running on http://localhost:8000.");
    }

    const contentType = res.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
        ? await res.json()
        : { detail: await res.text() };

    if (!res.ok) {
        const detail = Array.isArray(data.detail)
            ? data.detail.map((item) => item.msg).join(". ")
            : data.detail;
        throw new Error(detail || `Request failed with status ${res.status}`);
    }

    return data;
}

export function fetchAliases({ skip = 0, limit = 100, search = "", category = "", is_active = "" } = {}) {
    const params = new URLSearchParams({ skip, limit });
    if (search) params.append("search", search);
    if (category) params.append("category", category);
    if (is_active !== "") params.append("is_active", is_active);
    return requestAlias(`/?${params.toString()}`);
}

export function createAlias(data) {
    return requestAlias("/", {
        method: "POST",
        body: JSON.stringify(data),
    });
}

export function updateAlias(id, data) {
    return requestAlias(`/${encodeURIComponent(id)}`, {
        method: "PUT",
        body: JSON.stringify(data),
    });
}

export function deleteAlias(id) {
    return requestAlias(`/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function importAliases(file) {
    const formData = new FormData();
    formData.append("file", file);
    return requestAlias("/import", {
        method: "POST",
        body: formData,
    });
}
