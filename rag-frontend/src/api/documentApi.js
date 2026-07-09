import { API_BASE } from "./chatApi";
import { getAdminToken } from "../utils/adminAuth";

export async function uploadDocument(file, onProgress, onStatus, strategy = "auto") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("strategy", strategy);

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
            } catch (e) {}
            throw new Error(errorMsg);
        }

        // Read NDJSON stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let finalData = null;

        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\\n').filter(Boolean);
                for (const line of lines) {
                    try {
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
                    } catch (e) {
                        if (e.message !== "Unexpected end of JSON input") {
                            // If it's the error thrown by `data.error`, re-throw it
                            if (line.includes('"error"')) throw e;
                        }
                    }
                }
            }
        }
        return finalData || { message: "Document processed successfully" };
    } catch (error) {
        throw new Error(error.message || "Document upload failed. Make sure the backend server is running.");
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

export function getDocuments() {
    return requestDocument("/documents/");
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
