import { API_BASE } from "./chatApi";
import { getAdminToken } from "../utils/adminAuth";

export async function uploadDocument(file, onProgress, strategy = "auto") {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("strategy", strategy);

    return new Promise((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open("POST", `${API_BASE}/api/admin/documents/`);

        const token = getAdminToken();
        if (token) {
            request.setRequestHeader("Authorization", `Bearer ${token}`);
        }

        request.upload.onprogress = (event) => {
            if (!event.lengthComputable) return;
            const nextProgress = Math.round((event.loaded / event.total) * 80);
            onProgress?.(Math.max(8, nextProgress));
        };

        request.onload = () => {
            let data;

            try {
                data = JSON.parse(request.responseText);
            } catch {
                data = { detail: request.responseText };
            }

            if (request.status >= 200 && request.status < 300) {
                onProgress?.(100);
                resolve(data);
                return;
            }

            reject(new Error(data.detail || "Document upload failed"));
        };

        request.onerror = () => reject(new Error("Document upload failed. Make sure the backend server is running."));
        request.send(formData);
    });
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
