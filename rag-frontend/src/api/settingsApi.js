import { API_BASE, BACKEND_UNAVAILABLE_MESSAGE } from "./chatApi";
import { getAdminToken } from "../utils/adminAuth";

async function requestSettings(path, options = {}) {
    let res;

    try {
        res = await fetch(`${API_BASE}/api/admin${path}`, {
            headers: {
                "Content-Type": "application/json",
                ...(getAdminToken() ? { Authorization: `Bearer ${getAdminToken()}` } : {}),
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
        throw new Error(data.detail || `Request failed with status ${res.status}`);
    }

    return data;
}

/** GET /api/admin/settings/ */
export function getSettings() {
    return requestSettings("/settings/");
}

/** PUT /api/admin/settings/{key} */
export function updateSetting(key, value) {
    return requestSettings(`/settings/${encodeURIComponent(key)}`, {
        method: "PUT",
        body: JSON.stringify({ value: String(value) }),
    });
}

/** POST /api/admin/documents/reindex-all */
export function reindexAll(force = false) {
    return requestSettings(`/documents/reindex-all${force ? "?force=true" : ""}`, { method: "POST" });
}

export function getCrawlerStatus() {
    return requestSettings("/crawler/status");
}

export function triggerFullCrawler() {
    return requestSettings("/crawler/trigger-full", { method: "POST" });
}

export function resumeFullCrawler() {
    return requestSettings("/crawler/resume-full", { method: "POST" });
}

export function triggerAnnouncementCrawler() {
    return requestSettings("/crawler/trigger-announcements", { method: "POST" });
}

export function cancelCrawler(jobType) {
    return requestSettings(`/crawler/cancel/${encodeURIComponent(jobType)}`, { method: "POST" });
}
