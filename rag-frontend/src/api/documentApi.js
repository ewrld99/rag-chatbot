import { API_BASE } from "./chatApi";
import { getAdminToken } from "../utils/adminAuth";

export async function uploadDocument(file, onProgress, onStatus, strategy = "auto", programme = null, year = null) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("strategy", strategy);
    if (programme) formData.append("programme", programme);
    if (year !== null && year !== undefined) formData.append("year", String(year));

    const headers = {};
    const token = getAdminToken();
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }

    try {
        const response = await fetch(`${API_BASE}/api/admin/documents/`, {
            method: "POST",
            headers,
            body: formData,
        });

        if (!response.ok) {
            let errorMsg = "Document upload failed";
            try {
                const errData = await response.json();
                errorMsg = errData.detail || errorMsg;
            } catch {
                // Keep the generic upload error when the backend response is not JSON.
            }
            throw new Error(errorMsg);
        }

        // Read NDJSON stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let finalData = null;
        let buffered = "";

        const consumeLine = (line) => {
            if (!line.trim()) return;
            const data = JSON.parse(line);
            if (data.error) {
                throw new Error(data.error);
            }
            if (data.progress !== undefined) {
                onProgress?.(data.progress);
            }
            if (data.status) {
                onStatus?.(data.status);
            }
            if (data.progress === 100 && data.chunks_stored !== undefined) {
                finalData = data;
            }
        };

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                buffered += decoder.decode(value, { stream: true });
                const lines = buffered.split("\n");
                buffered = lines.pop() || "";
                for (const line of lines) {
                    consumeLine(line);
                }
            }
        }
        buffered += decoder.decode();
        if (buffered.trim()) consumeLine(buffered);
        return finalData || { message: "Document processed successfully" };
    } catch (error) {
        throw new Error(
            error.message || "Document upload failed. Make sure the backend server is running.",
            { cause: error },
        );
    }
}

async function requestDocument(path, options = {}) {
    const res = await fetch(`${API_BASE}/api/admin${path}`, {
        headers: options.body instanceof FormData ? undefined : {
            "Content-Type": "application/json",
            ...(getAdminToken() ? { Authorization: `Bearer ${getAdminToken()}` } : {}),
        },
        ...options,
    });

    const data = await res.json();

    if (!res.ok) {
        throw new Error(data.detail || "Document request failed");
    }

    return data;
}

export function getDocuments(page = 1, limit = 50, search = "") {
    const params = new URLSearchParams({ page, limit });
    if (search) params.append("search", search);
    return requestDocument(`/documents/?${params.toString()}`);
}

export function getDocument(id) {
    return requestDocument(`/documents/${encodeURIComponent(id)}`);
}

export function updateDocument(id, payload) {
    return requestDocument(`/documents/${encodeURIComponent(id)}`, {
        method: "PUT",
        body: JSON.stringify(payload),
    });
}

export function deleteDocument(id) {
    return requestDocument(`/documents/${encodeURIComponent(id)}`, {
        method: "DELETE",
    });
}

export function deleteDocumentsBatch(ids) {
    return requestDocument(`/documents/delete-batch`, {
        method: "POST",
        body: JSON.stringify(ids),
    });
}

export function reindexDocument(id) {
    return requestDocument(`/documents/${encodeURIComponent(id)}/reindex`, {
        method: "POST",
    });
}

export function getDocumentQuality(id) {
    return requestDocument(`/documents/${encodeURIComponent(id)}/quality`);
}

export function auditDocumentQuality(ids = []) {
    return requestDocument(`/documents/quality-audit`, {
        method: "POST",
        body: JSON.stringify({ document_ids: ids }),
    });
}

export function approveDocumentQuality(id, reason) {
    return requestDocument(`/documents/${encodeURIComponent(id)}/quality/approve`, {
        method: "POST",
        body: JSON.stringify({ reason }),
    });
}
