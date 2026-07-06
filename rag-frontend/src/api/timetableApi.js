import { API_BASE } from "./chatApi";
import { getAdminToken } from "../utils/adminAuth";

async function requestTimetable(path, options = {}) {
    const res = await fetch(`${API_BASE}/api/admin/timetable${path}`, {
        headers: {
            "Content-Type": "application/json",
            ...(getAdminToken() ? { Authorization: `Bearer ${getAdminToken()}` } : {}),
        },
        ...options,
    });

    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.detail || "Timetable request failed");
    }
    return data;
}

export function getYears() {
    return requestTimetable("/years");
}

export function getSemesters(year) {
    return requestTimetable(`/semesters?year=${encodeURIComponent(year)}`);
}

export function getCategories(year, semester) {
    return requestTimetable(
        `/categories?year=${encodeURIComponent(year)}&semester=${encodeURIComponent(semester)}`
    );
}

export function getOptionTypes(year, semester, category) {
    return requestTimetable(
        `/option-types?year=${encodeURIComponent(year)}&semester=${encodeURIComponent(semester)}&category=${encodeURIComponent(category)}`
    );
}

export function getDataOptions(year, semester, category, option) {
    return requestTimetable(
        `/data-options?year=${encodeURIComponent(year)}&semester=${encodeURIComponent(semester)}&category=${encodeURIComponent(category)}&option=${encodeURIComponent(option)}`
    );
}

export function fetchTimetable(payload) {
    return requestTimetable("/fetch", {
        method: "POST",
        body: JSON.stringify(payload),
    });
}

export function getFetchProgress(taskId) {
    return requestTimetable(`/progress/${encodeURIComponent(taskId)}`);
}
