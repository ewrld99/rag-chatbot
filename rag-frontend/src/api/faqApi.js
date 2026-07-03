import { getAdminSession } from "../utils/adminAuth.js";

const API_BASE = "http://localhost:8000/api/admin/faqs";

function getHeaders() {
    const session = getAdminSession();
    return {
        "Content-Type": "application/json",
        Authorization: session?.token ? `Bearer ${session.token}` : "",
    };
}

export async function fetchFaqs({ skip = 0, limit = 100, search = "", category = "", is_active = "" }) {
    const params = new URLSearchParams({ skip, limit });
    if (search) params.append("search", search);
    if (category) params.append("category", category);
    if (is_active !== "") params.append("is_active", is_active);

    const res = await fetch(`${API_BASE}/?${params.toString()}`, {
        headers: getHeaders(),
    });
    if (!res.ok) throw new Error("Failed to fetch FAQs");
    return res.json();
}

export async function createFaq(data) {
    const res = await fetch(`${API_BASE}/`, {
        method: "POST",
        headers: getHeaders(),
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to create FAQ");
    return res.json();
}

export async function updateFaq(id, data) {
    const res = await fetch(`${API_BASE}/${id}`, {
        method: "PUT",
        headers: getHeaders(),
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to update FAQ");
    return res.json();
}

export async function deleteFaq(id) {
    const res = await fetch(`${API_BASE}/${id}`, {
        method: "DELETE",
        headers: getHeaders(),
    });
    if (!res.ok) throw new Error("Failed to delete FAQ");
    return res.json();
}

export async function importFaqs(file) {
    const formData = new FormData();
    formData.append("file", file);

    const session = getAdminSession();
    const res = await fetch(`${API_BASE}/import`, {
        method: "POST",
        headers: {
            Authorization: session?.token ? `Bearer ${session.token}` : "",
        },
        body: formData,
    });
    if (!res.ok) throw new Error("Failed to import FAQs");
    return res.json();
}
