import { useCallback, useEffect, useMemo, useState } from "react";
import { deleteDocument, getDocuments, getDocument, updateDocument, deleteDocumentsBatch, reindexDocument } from "../../api/documentApi";
import { reindexAll } from "../../api/settingsApi";

const emptyForm = { content: "", filename: "" };

const RefreshIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" />
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
    </svg>
);
const EditIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
);
const TrashIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="3 6 5 6 21 6" />
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
);
const SaveIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
        <polyline points="17 21 17 13 7 13 7 21" /><polyline points="7 3 7 8 15 8" />
    </svg>
);
const XIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
);
const FileTextIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
        <path d="M14 2v4a2 2 0 0 0 2 2h4" /><path d="M10 9H8" /><path d="M16 13H8" /><path d="M16 17H8" />
    </svg>
);
const CheckIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12" />
    </svg>
);
const AlertIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
);
const InboxIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 16 12 14 15 10 15 8 12 2 12" />
        <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
    </svg>
);
const ReindexIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" />
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
    </svg>
);

const STATUS_CONFIG = {
    active:         { label: "Indexed",       bg: "#e8f5e9", color: "#2e7d32", border: "#a5d6a7" },
    needs_reindex:  { label: "Needs Reindex", bg: "#fff8e1", color: "#856404", border: "#f5d878" },
    processing:     { label: "Processing",    bg: "#e3f2fd", color: "#1565c0", border: "#90caf9" },
    failed:         { label: "Failed",        bg: "#ffebee", color: "#c62828", border: "#ef9a9a" },
};

function StatusBadge({ status }) {
    const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.active;
    return (
        <span style={{
            fontSize: "11px",
            fontWeight: "700",
            padding: "2px 8px",
            borderRadius: "20px",
            background: cfg.bg,
            color: cfg.color,
            border: `1px solid ${cfg.border}`,
            letterSpacing: "0.3px",
            whiteSpace: "nowrap",
        }}>
            {cfg.label}
        </span>
    );
}

function useMediaQuery(query) {
    const [matches, setMatches] = useState(() => (
        typeof window !== "undefined" ? window.matchMedia(query).matches : false
    ));

    useEffect(() => {
        const mediaQuery = window.matchMedia(query);
        const handleChange = () => setMatches(mediaQuery.matches);

        handleChange();
        mediaQuery.addEventListener("change", handleChange);

        return () => mediaQuery.removeEventListener("change", handleChange);
    }, [query]);

    return matches;
}

export default function DocumentManager({ refreshKey = 0, onChanged }) {
    const isNarrow = useMediaQuery("(max-width: 760px)");
    const [documents, setDocuments] = useState([]);
    const [page, setPage] = useState(1);
    const [limit] = useState(10);
    const [searchInput, setSearchInput] = useState("");
    const [searchQuery, setSearchQuery] = useState("");
    const [, setTotalDocs] = useState(0);
    const [totalPages, setTotalPages] = useState(1);
    const [form, setForm] = useState(emptyForm);
    const [editingId, setEditingId] = useState(null);
    const [status, setStatus] = useState("");
    const [statusType, setStatusType] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [isReindexing, setIsReindexing] = useState(false);
    const [focused, setFocused] = useState(null);
    const [selectedIds, setSelectedIds] = useState(new Set());

    const toggleSelection = (id) => {
        const newSet = new Set(selectedIds);
        if (newSet.has(id)) newSet.delete(id);
        else newSet.add(id);
        setSelectedIds(newSet);
    };

    const toggleSelectAll = () => {
        if (selectedIds.size === documents.length) {
            setSelectedIds(new Set());
        } else {
            setSelectedIds(new Set(documents.map(d => d.id)));
        }
    };

    const handleDeleteSelected = async () => {
        const count = selectedIds.size;
        if (!count) return;
        const confirmed = window.confirm(`Delete ${count} documents? This action cannot be undone.`);
        if (!confirmed) return;
        setIsLoading(true);
        try {
            await deleteDocumentsBatch(Array.from(selectedIds));
            setSelectedIds(new Set());
            if (selectedIds.has(editingId)) resetForm();
            setStatus(`${count} document(s) deleted.`);
            setStatusType("success");
            await loadDocuments({ clearStatus: false });
            onChanged?.();
        } catch (error) {
            setStatus(error.message);
            setStatusType("error");
        } finally {
            setIsLoading(false);
        }
    };

    const selectedDocument = useMemo(
        () => documents.find((d) => d.id === editingId),
        [documents, editingId]
    );
    const paginationItems = useMemo(() => {
        const maxVisible = isNarrow ? 3 : 5;
        const pages = new Set([1, totalPages, page]);
        const radius = Math.floor(maxVisible / 2);

        for (let offset = -radius; offset <= radius; offset += 1) {
            const candidate = page + offset;
            if (candidate >= 1 && candidate <= totalPages) {
                pages.add(candidate);
            }
        }

        const sorted = Array.from(pages).sort((a, b) => a - b);
        const items = [];
        sorted.forEach((value, index) => {
            if (index > 0 && value - sorted[index - 1] > 1) {
                items.push(`gap-${sorted[index - 1]}-${value}`);
            }
            items.push(value);
        });
        return items;
    }, [isNarrow, page, totalPages]);

    const loadDocuments = useCallback(async ({ clearStatus = true } = {}) => {
        setIsLoading(true);
        try {
            const data = await getDocuments(page, limit, searchQuery);
            const nextTotalPages = Math.max(1, Number(data.total_pages) || 1);
            setDocuments(data.items);
            setTotalDocs(data.total);
            setTotalPages(nextTotalPages);
            if (page > nextTotalPages) {
                setPage(nextTotalPages);
            }
            if (clearStatus) { setStatus(""); setStatusType(""); }
        } catch (error) {
            setStatus(error.message);
            setStatusType("error");
        } finally {
            setIsLoading(false);
        }
    }, [page, limit, searchQuery]);

    useEffect(() => {
        const timer = setTimeout(() => {
            setSearchQuery(searchInput);
            setPage(1);
        }, 300);
        return () => clearTimeout(timer);
    }, [searchInput]);

    useEffect(() => {
        const timer = window.setTimeout(() => loadDocuments(), 0);
        return () => window.clearTimeout(timer);
    }, [loadDocuments, refreshKey]);

    // Poll gently while any document is still processing.
    useEffect(() => {
        const hasProcessing = documents.some((d) => d.status === "processing");
        if (!hasProcessing) return undefined;
        const timer = window.setTimeout(() => {
            loadDocuments({ clearStatus: false });
        }, 7000);
        return () => {
            window.clearTimeout(timer);
        };
    }, [documents, loadDocuments]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setForm((c) => ({ ...c, [name]: value }));
    };

    const resetForm = () => { setForm(emptyForm); setEditingId(null); };

    const toPayload = () => {
        const payload = { content: form.content.trim() };
        const filename = form.filename.trim();
        if (filename) payload.filename = filename;
        return payload;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!form.content.trim()) {
            setStatus("Content is required.");
            setStatusType("error");
            return;
        }
        setIsSaving(true);
        try {
            await updateDocument(editingId, toPayload());
            setStatus("Document updated successfully.");
            setStatusType("success");
            resetForm();
            await loadDocuments({ clearStatus: false });
            onChanged?.();
        } catch (error) {
            setStatus(error.message);
            setStatusType("error");
        } finally {
            setIsSaving(false);
        }
    };

    const handleEdit = async (doc) => {
        setIsLoading(true);
        setStatus("");
        setStatusType("");
        try {
            const fullDoc = await getDocument(doc.id);
            setEditingId(doc.id);
            setForm({ content: fullDoc.content || "", filename: fullDoc.title || fullDoc.filename || "" });
        } catch (error) {
            setStatus(error.message);
            setStatusType("error");
        } finally {
            setIsLoading(false);
        }
    };

    const handleDelete = async (doc) => {
        const confirmed = window.confirm(
            `Delete "${doc.title || doc.filename || "this document"}"? This removes the whole document from search.`
        );
        if (!confirmed) return;
        try {
            await deleteDocument(doc.id);
            if (editingId === doc.id) resetForm();
            setStatus("Document deleted.");
            setStatusType("success");
            await loadDocuments({ clearStatus: false });
            onChanged?.();
        } catch (error) {
            setStatus(error.message);
            setStatusType("error");
        }
    };

    const handleReindexAll = async () => {
        setIsReindexing(true);
        try {
            const result = await reindexAll(!hasNeedsReindex);
            setStatus(result.message || "Reindex started.");
            setStatusType("success");
            await loadDocuments({ clearStatus: false });
            onChanged?.();
        } catch (error) {
            setStatus(error.message);
            setStatusType("error");
        } finally {
            setIsReindexing(false);
        }
    };

    const handleReindexSingle = async (doc) => {
        try {
            const result = await reindexDocument(doc.id);
            setStatus(result.message || "Reindex started.");
            setStatusType("success");
            await loadDocuments({ clearStatus: false });
            onChanged?.();
        } catch (error) {
            setStatus(error.message);
            setStatusType("error");
        }
    };

    const hasNeedsReindex = documents.some((d) => d.status === "needs_reindex");
    const isProcessing = documents.some((d) => d.status === "processing");
    const goToPage = (nextPage) => {
        const normalized = Number(nextPage);
        if (!Number.isFinite(normalized)) return;
        const maxPage = Math.max(1, totalPages);
        setPage(Math.min(maxPage, Math.max(1, Math.trunc(normalized))));
    };

    return (
        <section style={styles.page}>
            {/* Header */}
            <div style={{ ...styles.header, ...(isNarrow ? styles.headerNarrow : {}) }}>
                <div>
                    <span style={styles.eyebrow}>Library Controls</span>
                    <h2 style={styles.title}>Manage Documents</h2>
                </div>
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
                    <input
                        type="text"
                        placeholder="Search files..."
                        value={searchInput}
                        onChange={(e) => setSearchInput(e.target.value)}
                        style={styles.searchInput}
                    />
                    {selectedIds.size > 0 && (
                        <button
                            type="button"
                            style={{
                                ...styles.ghostBtn,
                                color: "var(--app-danger)",
                                borderColor: "var(--app-danger-soft)",
                                background: "var(--app-danger-soft)",
                                ...(isLoading ? styles.disabledBtn : {})
                            }}
                            onClick={handleDeleteSelected}
                            disabled={isLoading}
                        >
                            <TrashIcon />
                            Delete ({selectedIds.size})
                        </button>
                    )}
                    <button
                        type="button"
                        style={{
                            ...styles.reindexBtn,
                            ...((isReindexing || isProcessing) ? styles.disabledBtn : {}),
                            ...(hasNeedsReindex ? { background: "#ffe0b2", borderColor: "#ffb74d", color: "#e65100" } : {})
                        }}
                        onClick={handleReindexAll}
                        disabled={isReindexing || isProcessing}
                        id="btn-reindex-all"
                    >
                        <ReindexIcon />
                        {isReindexing || isProcessing
                            ? "Reindexing…" 
                            : hasNeedsReindex ? "Reindex Pending" : "Reindex All"}
                    </button>
                    <button
                        type="button"
                        style={{
                            ...styles.ghostBtn,
                            ...(isNarrow ? styles.fullWidthButton : {}),
                            ...(isLoading ? styles.disabledBtn : {}),
                        }}
                        onClick={loadDocuments}
                        disabled={isLoading}
                    >
                        <RefreshIcon />
                    </button>
                </div>
            </div>

            {/* Edit Panel */}
            {editingId ? (
                <div style={{ ...styles.editPanel, ...(isNarrow ? styles.editPanelNarrow : {}) }}>
                    <div style={{ ...styles.editPanelHeader, ...(isNarrow ? styles.editPanelHeaderNarrow : {}) }}>
                        <div style={styles.editHeaderText}>
                            <span style={styles.eyebrow}>Edit Mode</span>
                            <h3 style={{ ...styles.editTitle, ...(isNarrow ? styles.editTitleNarrow : {}) }}>
                                {selectedDocument?.filename || selectedDocument?.title || "Manual Entry"}
                                {selectedDocument && (
                                    <span style={styles.chunksBadge}>
                                        {selectedDocument.chunk_count ?? 1} parts
                                    </span>
                                )}
                            </h3>
                        </div>
                        <button type="button" style={styles.closeBtn} onClick={resetForm} aria-label="Cancel edit">
                            <XIcon />
                        </button>
                    </div>

                    <form style={styles.form} onSubmit={handleSubmit}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.label} htmlFor="dm-content">Content</label>
                            <textarea
                                id="dm-content"
                                name="content"
                                value={form.content}
                                onChange={handleChange}
                                onFocus={() => setFocused("content")}
                                onBlur={() => setFocused(null)}
                                placeholder="Edit the searchable text for this document"
                                rows="7"
                                style={{
                                    ...styles.textarea,
                                    ...(focused === "content" ? styles.inputFocused : {}),
                                }}
                            />
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label} htmlFor="dm-filename">Filename</label>
                            <input
                                id="dm-filename"
                                name="filename"
                                value={form.filename}
                                onChange={handleChange}
                                onFocus={() => setFocused("filename")}
                                onBlur={() => setFocused(null)}
                                placeholder="policy.pdf"
                                style={{
                                    ...styles.input,
                                    ...(focused === "filename" ? styles.inputFocused : {}),
                                }}
                            />
                        </div>

                        <div style={{ ...styles.formActions, ...(isNarrow ? styles.formActionsNarrow : {}) }}>
                            <button
                                type="submit"
                                style={{
                                    ...styles.primaryBtn,
                                    ...(isNarrow ? styles.fullWidthButton : {}),
                                    ...(isSaving ? styles.disabledBtn : {}),
                                }}
                                disabled={isSaving}
                            >
                                <SaveIcon />
                                {isSaving ? "Saving…" : "Update Document"}
                            </button>
                            <button
                                type="button"
                                style={{ ...styles.ghostBtn, ...(isNarrow ? styles.fullWidthButton : {}) }}
                                onClick={resetForm}
                            >
                                <XIcon />
                                Cancel
                            </button>
                        </div>
                    </form>
                </div>
            ) : (
                <div style={{ ...styles.emptyEditor, ...(isNarrow ? styles.emptyEditorNarrow : {}) }}>
                    <div style={styles.emptyEditorIcon}><EditIcon /></div>
                    <h3 style={styles.emptyEditorTitle}>Select a document to edit</h3>
                    <p style={styles.emptyEditorText}>Choose Edit on any document below to update its text or filename.</p>
                </div>
            )}

            {/* Status */}
            {status && (
                <div style={{
                    ...styles.statusBox,
                    ...(statusType === "success" ? styles.statusSuccess : statusType === "error" ? styles.statusError : {}),
                }}>
                    {statusType === "success" ? <CheckIcon /> : <AlertIcon />}
                    <span>{status}</span>
                </div>
            )}

            {/* Document List */}
            <div style={styles.docList}>
                {documents.length > 0 && !isLoading && (
                    <div style={{ display: "flex", alignItems: "center", padding: "0 18px", marginBottom: "8px", gap: "10px" }}>
                        <input
                            type="checkbox"
                            checked={documents.length > 0 && selectedIds.size === documents.length}
                            onChange={toggleSelectAll}
                            style={{ cursor: "pointer", width: "16px", height: "16px" }}
                        />
                        <span style={{ fontSize: "13px", color: "#5f594f", fontWeight: "600" }}>Select All</span>
                    </div>
                )}
                {documents.length === 0 && !isLoading ? (
                    <div style={styles.emptyList}>
                        <InboxIcon />
                        <p style={styles.emptyListText}>No documents yet. Use Upload to add PDF, DOCX, or TXT files.</p>
                    </div>
                ) : (
                    documents.map((doc) => (
                        <article
                            key={doc.id}
                            style={{
                                ...styles.docRow,
                                ...(editingId === doc.id ? styles.docRowActive : {}),
                                ...(isNarrow ? styles.docRowNarrow : {}),
                            }}
                        >
                            <div style={{ display: "flex", alignItems: "center", alignSelf: "flex-start", marginTop: "4px" }}>
                                <input
                                    type="checkbox"
                                    checked={selectedIds.has(doc.id)}
                                    onChange={() => toggleSelection(doc.id)}
                                    style={{ cursor: "pointer", width: "16px", height: "16px" }}
                                />
                            </div>
                            <div style={styles.docRowInfo}>
                                <div style={styles.docRowMeta}>
                                    <span style={styles.docRowSource}>
                                        <FileTextIcon />
                                        {doc.title || doc.filename || "Manual entry"}
                                    </span>
                                    <span style={styles.docRowChunks}>
                                        {doc.chunk_count ?? 1} parts
                                    </span>
                                    <StatusBadge status={doc.status || "active"} />
                                </div>
                                <p style={{ ...styles.docRowPreview, ...(isNarrow ? styles.docRowPreviewNarrow : {}) }}>{doc.content}</p>
                            </div>
                            <div style={{ ...styles.docRowActions, ...(isNarrow ? styles.docRowActionsNarrow : {}) }}>
                                <button
                                    type="button"
                                    style={{ 
                                        ...styles.ghostBtn, 
                                        ...(isNarrow ? styles.editBtnNarrow : {}), 
                                        padding: "6px 10px", 
                                        fontSize: "12px",
                                        ...(doc.status === "processing" ? styles.disabledBtn : {})
                                    }}
                                    onClick={() => handleReindexSingle(doc)}
                                    disabled={doc.status === "processing"}
                                >
                                    <ReindexIcon />
                                    {doc.status === "processing" ? "Reindexing…" : "Reindex"}
                                </button>
                                <button
                                    type="button"
                                    style={{ ...styles.editBtn, ...(isNarrow ? styles.editBtnNarrow : {}) }}
                                    onClick={() => handleEdit(doc)}
                                >
                                    <EditIcon />
                                    Edit
                                </button>
                                <button
                                    type="button"
                                    style={{ ...styles.deleteBtn, ...(isNarrow ? styles.deleteBtnNarrow : {}) }}
                                    onClick={() => handleDelete(doc)}
                                >
                                    <TrashIcon />
                                </button>
                            </div>
                        </article>
                    ))
                )}
            </div>

            {/* Pagination Controls */}
            <div style={{ ...styles.paginationBar, ...(isNarrow ? styles.paginationBarNarrow : {}) }}>
                <div style={styles.paginationContainer}>
                    <button
                        type="button"
                        style={{ ...styles.pageButton, ...(page === 1 ? styles.disabledBtn : {}) }}
                        onClick={() => goToPage(1)}
                        disabled={page === 1}
                    >
                        First
                    </button>
                    <button
                        type="button"
                        style={{ ...styles.pageButton, ...(page === 1 ? styles.disabledBtn : {}) }}
                        onClick={() => goToPage(page - 1)}
                        disabled={page === 1}
                    >
                        Previous
                    </button>
                    <div style={styles.pageNumberGroup}>
                        {paginationItems.map((item) => (
                            typeof item === "number" ? (
                                <button
                                    type="button"
                                    key={item}
                                    style={{
                                        ...styles.pageNumberButton,
                                        ...(item === page ? styles.pageNumberButtonActive : {}),
                                    }}
                                    onClick={() => goToPage(item)}
                                    aria-current={item === page ? "page" : undefined}
                                >
                                    {item}
                                </button>
                            ) : (
                                <span key={item} style={styles.pageGap}>...</span>
                            )
                        ))}
                    </div>
                    <button
                        type="button"
                        style={{ ...styles.pageButton, ...(page === totalPages ? styles.disabledBtn : {}) }}
                        onClick={() => goToPage(page + 1)}
                        disabled={page === totalPages}
                    >
                        Next
                    </button>
                    <button
                        type="button"
                        style={{ ...styles.pageButton, ...(page === totalPages ? styles.disabledBtn : {}) }}
                        onClick={() => goToPage(totalPages)}
                        disabled={page === totalPages}
                    >
                        Last
                    </button>
                </div>
                <form
                    style={styles.pageJumpForm}
                    onSubmit={(event) => {
                        event.preventDefault();
                        const formData = new FormData(event.currentTarget);
                        goToPage(formData.get("page"));
                        event.currentTarget.reset();
                    }}
                >
                    <span style={styles.pageText}>Page {page} of {totalPages}</span>
                    <input
                        name="page"
                        type="number"
                        min="1"
                        max={totalPages}
                        placeholder="Go"
                        style={styles.pageJumpInput}
                        aria-label="Go to page"
                    />
                </form>
            </div>
        </section>
    );
}

const styles = {
    page: {
        width: "100%",
        maxWidth: "100%",
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        gap: "20px",
        overflowX: "hidden",
    },
    header: {
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "space-between",
        gap: "16px",
    },
    headerNarrow: {
        flexDirection: "column",
        alignItems: "stretch",
    },
    eyebrow: {
        display: "block",
        fontSize: "11px",
        fontWeight: "700",
        letterSpacing: "1.5px",
        textTransform: "uppercase",
        color: "var(--app-accent)",
        marginBottom: "4px",
    },
    searchInput: {
        padding: "8px 12px",
        borderRadius: "8px",
        border: "1px solid var(--app-border-strong)",
        background: "var(--app-bg)",
        color: "var(--app-text)",
        fontSize: "13px",
        outline: "none",
        width: "200px",
    },
    paginationContainer: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "6px",
        flexWrap: "wrap",
    },
    paginationBar: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: "12px",
        marginTop: "16px",
        padding: "10px 14px",
        background: "var(--app-surface)",
        borderRadius: "12px",
        border: "1px solid var(--app-border)",
        flexWrap: "wrap",
    },
    paginationBarNarrow: {
        alignItems: "stretch",
        flexDirection: "column",
    },
    pageButton: {
        minHeight: "34px",
        padding: "7px 10px",
        borderRadius: "8px",
        border: "1px solid var(--app-border)",
        background: "var(--app-surface)",
        color: "var(--app-text)",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        transition: "all 0.15s",
    },
    pageNumberGroup: {
        display: "flex",
        alignItems: "center",
        gap: "4px",
        flexWrap: "wrap",
        justifyContent: "center",
    },
    pageNumberButton: {
        width: "34px",
        height: "34px",
        borderRadius: "8px",
        border: "1px solid var(--app-border)",
        background: "var(--app-bg)",
        color: "var(--app-text)",
        fontSize: "13px",
        fontWeight: "700",
        cursor: "pointer",
    },
    pageNumberButtonActive: {
        background: "var(--app-text)",
        color: "var(--app-surface)",
        borderColor: "var(--app-text)",
    },
    pageGap: {
        minWidth: "22px",
        textAlign: "center",
        color: "var(--app-muted)",
        fontSize: "13px",
        fontWeight: "700",
    },
    pageJumpForm: {
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-end",
        gap: "8px",
        flexWrap: "wrap",
    },
    pageJumpInput: {
        width: "72px",
        minHeight: "34px",
        borderRadius: "8px",
        border: "1px solid var(--app-border-strong)",
        background: "var(--app-bg)",
        color: "var(--app-text)",
        fontSize: "13px",
        padding: "6px 8px",
        boxSizing: "border-box",
        outline: "none",
    },
    pageText: {
        fontSize: "13px",
        color: "var(--app-muted)",
        fontWeight: "500",
    },
    title: { fontSize: "22px", fontWeight: "700", color: "var(--app-text)", margin: 0 },
    ghostBtn: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "6px",
        padding: "8px 14px",
        borderRadius: "10px",
        border: "1px solid var(--app-border-strong)",
        background: "transparent",
        color: "#4d4942",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
    },
    reindexBtn: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "6px",
        padding: "8px 14px",
        borderRadius: "10px",
        border: "1px solid #f5d878",
        background: "#fff8e1",
        color: "#856404",
        fontSize: "13px",
        fontWeight: "700",
        cursor: "pointer",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
    },
    primaryBtn: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "6px",
        padding: "10px 18px",
        borderRadius: "10px",
        border: "none",
        background: "var(--app-text)",
        color: "var(--app-surface-muted)",
        fontSize: "13px",
        fontWeight: "700",
        cursor: "pointer",
    },
    disabledBtn: { opacity: 0.5, cursor: "not-allowed" },
    fullWidthButton: { width: "100%" },
    editPanel: {
        minWidth: 0,
        maxWidth: "100%",
        boxSizing: "border-box",
        background: "var(--app-surface)",
        border: "1px solid #e8cdb8",
        borderRadius: "16px",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        gap: "20px",
    },
    editPanelNarrow: {
        padding: "16px",
        borderRadius: "12px",
        gap: "16px",
    },
    editPanelHeader: {
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "space-between",
        gap: "12px",
    },
    editPanelHeaderNarrow: {
        alignItems: "flex-start",
    },
    editHeaderText: {
        minWidth: 0,
    },
    editTitle: {
        fontSize: "16px",
        fontWeight: "700",
        color: "var(--app-text)",
        margin: 0,
        display: "flex",
        alignItems: "center",
        gap: "8px",
        minWidth: 0,
        overflowWrap: "anywhere",
    },
    editTitleNarrow: {
        alignItems: "flex-start",
        flexDirection: "column",
    },
    chunksBadge: {
        fontSize: "11px",
        fontWeight: "600",
        color: "var(--app-accent)",
        background: "var(--app-accent-soft)",
        padding: "2px 8px",
        borderRadius: "10px",
    },
    closeBtn: {
        width: "32px",
        height: "32px",
        borderRadius: "8px",
        border: "1px solid var(--app-border-strong)",
        background: "transparent",
        color: "var(--app-muted)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        flexShrink: 0,
    },
    form: { display: "flex", flexDirection: "column", gap: "16px" },
    fieldGroup: { minWidth: 0, display: "flex", flexDirection: "column", gap: "6px" },
    label: { fontSize: "13px", fontWeight: "600", color: "#5f594f" },
    input: {
        width: "100%",
        background: "#ffffff",
        border: "1px solid var(--app-border-strong)",
        borderRadius: "10px",
        padding: "11px 14px",
        fontSize: "14px",
        color: "var(--app-text)",
        outline: "none",
        transition: "border-color 0.2s, box-shadow 0.2s",
        boxSizing: "border-box",
    },
    textarea: {
        width: "100%",
        maxWidth: "100%",
        background: "#ffffff",
        border: "1px solid var(--app-border-strong)",
        borderRadius: "10px",
        padding: "12px 14px",
        fontSize: "13px",
        color: "var(--app-text)",
        outline: "none",
        resize: "vertical",
        lineHeight: 1.6,
        transition: "border-color 0.2s, box-shadow 0.2s",
        fontFamily: "monospace",
        boxSizing: "border-box",
    },
    inputFocused: {
        borderColor: "var(--app-accent-strong)",
        boxShadow: "0 0 0 3px rgba(217,108,71,0.12)",
    },
    formActions: { display: "flex", gap: "10px", flexWrap: "wrap" },
    formActionsNarrow: { flexDirection: "column" },
    statusBox: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "11px 14px",
        borderRadius: "10px",
        fontSize: "13px",
    },
    statusSuccess: {
        background: "var(--app-accent-soft)",
        border: "1px solid #e8cdb8",
        color: "var(--app-accent)",
    },
    statusError: {
        background: "var(--app-danger-soft)",
        border: "1px solid #f1c4b2",
        color: "var(--app-danger)",
    },
    emptyEditor: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "8px",
        padding: "28px",
        borderRadius: "14px",
        border: "1px dashed var(--app-border-strong)",
        background: "var(--app-surface-muted)",
        textAlign: "center",
    },
    emptyEditorNarrow: {
        padding: "20px 14px",
    },
    emptyEditorIcon: {
        width: "40px",
        height: "40px",
        borderRadius: "10px",
        background: "var(--app-bg)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--app-faint)",
    },
    emptyEditorTitle: { fontSize: "15px", fontWeight: "600", color: "#4d4942", margin: 0 },
    emptyEditorText: { fontSize: "13px", color: "var(--app-faint)", margin: 0, maxWidth: "360px" },
    docList: { display: "flex", flexDirection: "column", gap: "10px" },
    docRow: {
        width: "100%",
        maxWidth: "100%",
        minWidth: 0,
        boxSizing: "border-box",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        gap: "16px",
        padding: "16px 18px",
        borderRadius: "14px",
        background: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        transition: "border-color 0.15s",
    },
    docRowNarrow: {
        flexDirection: "column",
        gap: "12px",
        padding: "14px",
    },
    docRowActive: {
        border: "1px solid #e8cdb8",
        background: "var(--app-accent-soft)",
    },
    docRowInfo: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        minWidth: 0,
        maxWidth: "100%",
        flex: "1 1 auto",
        overflow: "hidden",
    },
    docRowMeta: {
        minWidth: 0,
        display: "flex",
        alignItems: "center",
        gap: "10px",
        flexWrap: "wrap",
    },
    docRowSource: {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        fontSize: "13px",
        fontWeight: "600",
        color: "#4d4942",
        minWidth: 0,
        maxWidth: "100%",
        overflowWrap: "anywhere",
        wordBreak: "break-word",
    },
    docRowChunks: {
        fontSize: "11px",
        fontWeight: "600",
        color: "var(--app-accent)",
        background: "var(--app-accent-soft)",
        padding: "2px 8px",
        borderRadius: "10px",
    },
    docRowPreview: {
        fontSize: "12px",
        color: "var(--app-faint)",
        margin: 0,
        maxWidth: "100%",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        overflowWrap: "anywhere",
        wordBreak: "break-word",
    },
    docRowPreviewNarrow: {
        display: "-webkit-box",
        whiteSpace: "normal",
        WebkitLineClamp: 3,
        WebkitBoxOrient: "vertical",
        overflowWrap: "anywhere",
    },
    docRowActions: {
        display: "flex",
        gap: "6px",
        flex: "0 0 auto",
        alignItems: "center",
        maxWidth: "100%",
    },
    docRowActionsNarrow: {
        width: "100%",
    },
    editBtn: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "5px",
        padding: "6px 12px",
        borderRadius: "8px",
        border: "1px solid #e8cdb8",
        background: "var(--app-accent-soft)",
        color: "var(--app-accent)",
        fontSize: "12px",
        fontWeight: "600",
        cursor: "pointer",
    },
    editBtnNarrow: {
        flex: 1,
        minHeight: "38px",
    },
    deleteBtn: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "32px",
        height: "32px",
        borderRadius: "8px",
        border: "1px solid #f1c4b2",
        background: "var(--app-danger-soft)",
        color: "var(--app-danger)",
        cursor: "pointer",
    },
    deleteBtnNarrow: {
        width: "42px",
        height: "38px",
    },
    emptyList: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "12px",
        padding: "48px 20px",
        borderRadius: "14px",
        border: "1px dashed var(--app-border-strong)",
        color: "var(--app-faint)",
        textAlign: "center",
    },
    emptyListText: { fontSize: "13px", color: "var(--app-faint)", margin: 0, maxWidth: "360px" },
};
