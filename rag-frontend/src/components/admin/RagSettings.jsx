import { useCallback, useEffect, useRef, useState } from "react";
import { getSettings, updateSetting } from "../../api/settingsApi";

// ─── Icons ────────────────────────────────────────────────────────────────────
const SaveIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" /><polyline points="17 21 17 13 7 13 7 21" /><polyline points="7 3 7 8 15 8" />
    </svg>
);
const AlertTriangleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
);
const CheckIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12" />
    </svg>
);
const RefreshIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" />
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
    </svg>
);

// ─── Setting metadata ──────────────────────────────────────────────────────────
const SETTING_META = {
    chunk_size:         { label: "Chunk Size",           type: "number", min: 500,  max: 2000, step: 50,  reindex: true },
    chunk_overlap:      { label: "Chunk Overlap",        type: "number", min: 0,    max: 500,  step: 10,  reindex: true },
    top_k_dense:        { label: "Dense Top K",          type: "number", min: 1,    max: 50,   step: 1,   reindex: false },
    top_k_sparse:       { label: "Sparse Top K",         type: "number", min: 1,    max: 50,   step: 1,   reindex: false },
    top_k_final:        { label: "Final Top K",          type: "number", min: 1,    max: 20,   step: 1,   reindex: false },
    rrf_k:              { label: "RRF Constant",         type: "number", min: 10,   max: 200,  step: 5,   reindex: false },
    enable_reranker:    { label: "Enable Reranker",      type: "toggle",                                  reindex: false },
    embedding_model:    { label: "Embedding Model",      type: "select", 
                          options: [
                              'jina-embeddings-v2-base-en', 'jina-embeddings-v2-base-zh', 'jina-embeddings-v2-base-de', 
                              'jina-embeddings-v2-base-es', 'jina-embeddings-v2-base-code', 'jina-embeddings-v3', 
                              'jina-embeddings-v5-text-nano', 'jina-embeddings-v5-text-small', 'jina-embeddings-v5-omni-small', 
                              'jina-embeddings-v5-omni-nano', 'jina-embeddings-v4', 'jina-code-embeddings-0.5b', 
                              'jina-code-embeddings-1.5b', 'jina-clip-v1', 'jina-clip-v2', 'jina-colbert-v1-en', 
                              'jina-colbert-v2', 'elser-v2'
                          ], reindex: true  },
    max_upload_size_mb: { label: "Max Upload Size (MB)", type: "number", min: 1,    max: 100,  step: 1,   reindex: false },
    allowed_extensions: { label: "Allowed Extensions",  type: "text",                                    reindex: false },
};

const GROUPS = [
    {
        id: "rag",
        label: "RAG",
        eyebrow: "Chunking",
        description: "Controls how documents are split into searchable pieces.",
        keys: ["chunk_size", "chunk_overlap"],
    },
    {
        id: "retrieval",
        label: "Retrieval",
        eyebrow: "Hybrid Search",
        description: "Tune dense + sparse retrieval and Reciprocal Rank Fusion.",
        keys: ["top_k_dense", "top_k_sparse", "top_k_final", "rrf_k", "enable_reranker"],
    },
    {
        id: "embeddings",
        label: "Embeddings",
        eyebrow: "Embedding Model",
        description: "Which model generates vector representations of text.",
        keys: ["embedding_model"],
    },
    {
        id: "upload",
        label: "Upload",
        eyebrow: "File Ingestion",
        description: "Constraints on file uploads accepted by the system.",
        keys: ["max_upload_size_mb", "allowed_extensions"],
    },
];

// ─── Toast ─────────────────────────────────────────────────────────────────────
function Toast({ toasts }) {
    return (
        <div style={styles.toastContainer}>
            {toasts.map((t) => (
                <div key={t.id} style={{ ...styles.toast, ...(t.type === "success" ? styles.toastSuccess : styles.toastError) }}>
                    {t.type === "success" ? <CheckIcon /> : <AlertTriangleIcon />}
                    <span style={{ fontSize: "13px" }}>{t.message}</span>
                </div>
            ))}
        </div>
    );
}

// ─── Individual setting row ────────────────────────────────────────────────────
function SettingRow({ setting, meta, onSaved }) {
    const [value, setValue] = useState(setting.value ?? "");
    const [saving, setSaving] = useState(false);
    const [localError, setLocalError] = useState("");
    const originalValue = useRef(setting.value ?? "");

    // Sync when parent refreshes
    useEffect(() => {
        setValue(setting.value ?? "");
        originalValue.current = setting.value ?? "";
    }, [setting.value]);

    const validate = (v) => {
        if (!meta) return "";
        if (meta.type === "number") {
            const n = Number(v);
            if (isNaN(n) || !Number.isInteger(n)) return `Must be a whole number`;
            if (meta.min !== undefined && n < meta.min) return `Min is ${meta.min}`;
            if (meta.max !== undefined && n > meta.max) return `Max is ${meta.max}`;
        }
        if (meta.type === "text" && !v.trim()) return "Cannot be empty";
        return "";
    };

    const handleSave = async () => {
        const err = validate(value);
        if (err) { setLocalError(err); return; }
        setLocalError("");
        setSaving(true);
        try {
            await updateSetting(setting.key, value);
            originalValue.current = value;
            onSaved(setting.key, value, meta?.reindex ? "reindex" : "ok");
        } catch (e) {
            onSaved(setting.key, value, "error", e.message);
        } finally {
            setSaving(false);
        }
    };

    const dirty = value !== originalValue.current;

    const renderInput = () => {
        if (meta?.type === "toggle") {
            const on = value === "true";
            return (
                <button
                    type="button"
                    id={`setting-${setting.key}`}
                    style={{ ...styles.toggle, ...(on ? styles.toggleOn : styles.toggleOff) }}
                    onClick={() => setValue(on ? "false" : "true")}
                    aria-pressed={on}
                >
                    <span style={{ ...styles.toggleThumb, ...(on ? styles.toggleThumbOn : {}) }} />
                </button>
            );
        }
        if (meta?.type === "select") {
            return (
                <select
                    id={`setting-${setting.key}`}
                    value={value}
                    onChange={(e) => { setValue(e.target.value); setLocalError(""); }}
                    style={{ ...styles.input, ...(localError ? styles.inputError : dirty ? styles.inputDirty : {}) }}
                >
                    {meta.options.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                    ))}
                </select>
            );
        }
        return (
            <input
                id={`setting-${setting.key}`}
                type={meta?.type === "number" ? "number" : "text"}
                value={value}
                min={meta?.min}
                max={meta?.max}
                step={meta?.step}
                onChange={(e) => { setValue(e.target.value); setLocalError(""); }}
                style={{ ...styles.input, ...(localError ? styles.inputError : dirty ? styles.inputDirty : {}) }}
            />
        );
    };

    return (
        <div style={styles.settingRow}>
            <div style={styles.settingInfo}>
                <div style={styles.settingLabelRow}>
                    <label htmlFor={`setting-${setting.key}`} style={styles.settingLabel}>
                        {meta?.label || setting.key}
                    </label>
                    {meta?.reindex && (
                        <span style={styles.reindexBadge} title="Changing this setting requires reindexing all documents">
                            <AlertTriangleIcon /> Requires Reindex
                        </span>
                    )}
                </div>
                {setting.description && (
                    <p style={styles.settingDescription}>{setting.description}</p>
                )}
                {localError && <p style={styles.errorText}>{localError}</p>}
            </div>
            <div style={styles.settingControl}>
                {renderInput()}
                <button
                    type="button"
                    style={{
                        ...styles.saveBtn,
                        ...(saving || !dirty ? styles.saveBtnDisabled : styles.saveBtnActive),
                    }}
                    onClick={handleSave}
                    disabled={saving || !dirty}
                    title="Save this setting"
                >
                    {saving ? <RefreshIcon /> : <SaveIcon />}
                    {saving ? "Saving…" : "Save"}
                </button>
            </div>
        </div>
    );
}

// ─── Group card ────────────────────────────────────────────────────────────────
function SettingGroup({ group, settingMap, onSaved }) {
    const groupSettings = group.keys
        .map((k) => settingMap[k])
        .filter(Boolean);

    if (groupSettings.length === 0) return null;

    return (
        <article style={styles.card}>
            <div style={styles.cardHeader}>
                <span style={styles.eyebrow}>{group.eyebrow}</span>
                <h3 style={styles.cardTitle}>{group.label}</h3>
                <p style={styles.cardDescription}>{group.description}</p>
            </div>
            <div style={styles.settingList}>
                {groupSettings.map((s, i) => (
                    <div key={s.key}>
                        {i > 0 && <div style={styles.divider} />}
                        <SettingRow
                            setting={s}
                            meta={SETTING_META[s.key]}
                            onSaved={onSaved}
                        />
                    </div>
                ))}
            </div>
        </article>
    );
}

// ─── Main component ────────────────────────────────────────────────────────────
let toastId = 0;

export default function RagSettings() {
    const [settings, setSettings] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [loadError, setLoadError] = useState("");
    const [toasts, setToasts] = useState([]);

    const addToast = useCallback((message, type = "success") => {
        const id = ++toastId;
        setToasts((prev) => [...prev, { id, message, type }]);
        setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000);
    }, []);

    const loadSettings = useCallback(async () => {
        setIsLoading(true);
        setLoadError("");
        try {
            const data = await getSettings();
            setSettings(data);
        } catch (e) {
            setLoadError(e.message);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { loadSettings(); }, [loadSettings]);

    const handleSaved = useCallback((key, _value, outcome, errMsg) => {
        if (outcome === "error") {
            addToast(errMsg || "Failed to save setting.", "error");
            return;
        }
        if (outcome === "reindex") {
            addToast(
                `"${SETTING_META[key]?.label || key}" saved. All documents are now marked for reindexing.`,
                "success"
            );
            // Reload to reflect fresh DB values
            loadSettings();
            return;
        }
        addToast(`"${SETTING_META[key]?.label || key}" saved successfully.`, "success");
    }, [addToast, loadSettings]);

    const settingMap = Object.fromEntries(settings.map((s) => [s.key, s]));

    return (
        <section style={styles.page}>
            <Toast toasts={toasts} />

            {/* Page header */}
            <div style={styles.pageHeader}>
                <div>
                    <span style={styles.eyebrow}>Configuration</span>
                    <h2 style={styles.pageTitle}>RAG Settings</h2>
                    <p style={styles.pageSubtitle}>
                        Configure the retrieval-augmented generation pipeline. Changes take effect immediately.
                    </p>
                </div>
                <button
                    type="button"
                    style={{ ...styles.ghostBtn, ...(isLoading ? styles.ghostBtnDisabled : {}) }}
                    onClick={loadSettings}
                    disabled={isLoading}
                >
                    <RefreshIcon />
                    {isLoading ? "Loading…" : "Refresh"}
                </button>
            </div>

            {loadError && (
                <div style={styles.errorBanner}>
                    <AlertTriangleIcon />
                    <span>{loadError}</span>
                </div>
            )}

            {isLoading && !loadError && (
                <div style={styles.skeleton}>
                    {[1, 2, 3, 4].map((i) => (
                        <div key={i} style={styles.skeletonCard} />
                    ))}
                </div>
            )}

            {!isLoading && !loadError && (
                <div style={styles.groupGrid}>
                    {GROUPS.map((group) => (
                        <SettingGroup
                            key={group.id}
                            group={group}
                            settingMap={settingMap}
                            onSaved={handleSaved}
                        />
                    ))}
                </div>
            )}
        </section>
    );
}

// ─── Styles ────────────────────────────────────────────────────────────────────
const styles = {
    page: {
        display: "flex",
        flexDirection: "column",
        gap: "24px",
        width: "100%",
        maxWidth: "100%",
    },
    pageHeader: {
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "space-between",
        gap: "16px",
        flexWrap: "wrap",
    },
    eyebrow: {
        display: "block",
        fontSize: "11px",
        fontWeight: "700",
        letterSpacing: "1.5px",
        textTransform: "uppercase",
        color: "#9a4f35",
        marginBottom: "4px",
    },
    pageTitle: {
        fontSize: "22px",
        fontWeight: "700",
        color: "var(--text-primary, #2b2925)",
        margin: "0 0 6px",
    },
    pageSubtitle: {
        fontSize: "13px",
        color: "var(--text-muted, #8a8478)",
        margin: 0,
        maxWidth: "480px",
    },
    ghostBtn: {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        padding: "8px 14px",
        borderRadius: "10px",
        border: "1px solid #d7d0c1",
        background: "transparent",
        color: "#4d4942",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
    },
    ghostBtnDisabled: {
        opacity: 0.5,
        cursor: "not-allowed",
    },
    errorBanner: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "12px 16px",
        borderRadius: "10px",
        background: "#fff0e8",
        border: "1px solid #f1c4b2",
        color: "#a13f24",
        fontSize: "13px",
    },
    groupGrid: {
        display: "flex",
        flexDirection: "column",
        gap: "20px",
    },
    card: {
        background: "var(--card-bg, #fffdf8)",
        border: "1px solid var(--card-border, #ded9cd)",
        borderRadius: "16px",
        overflow: "hidden",
        boxShadow: "0 4px 16px rgba(72, 61, 47, 0.06)",
    },
    cardHeader: {
        padding: "22px 24px 16px",
        borderBottom: "1px solid var(--card-border, #ede8de)",
    },
    cardTitle: {
        fontSize: "16px",
        fontWeight: "700",
        color: "var(--text-primary, #2b2925)",
        margin: "0 0 4px",
    },
    cardDescription: {
        fontSize: "12px",
        color: "var(--text-muted, #8a8478)",
        margin: 0,
    },
    settingList: {
        padding: "4px 0",
    },
    settingRow: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "24px",
        padding: "16px 24px",
        flexWrap: "wrap",
    },
    settingInfo: {
        flex: "1 1 260px",
        minWidth: 0,
    },
    settingLabelRow: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        flexWrap: "wrap",
        marginBottom: "4px",
    },
    settingLabel: {
        fontSize: "14px",
        fontWeight: "600",
        color: "var(--text-primary, #2b2925)",
        cursor: "pointer",
    },
    reindexBadge: {
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        fontSize: "11px",
        fontWeight: "600",
        color: "#9a4f35",
        background: "#fef3cd",
        border: "1px solid #f5d878",
        padding: "2px 7px",
        borderRadius: "20px",
    },
    settingDescription: {
        fontSize: "12px",
        color: "var(--text-muted, #8a8478)",
        margin: 0,
        lineHeight: 1.5,
    },
    errorText: {
        fontSize: "12px",
        color: "#a13f24",
        margin: "4px 0 0",
    },
    settingControl: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
        flex: "0 0 auto",
    },
    input: {
        width: "180px",
        padding: "9px 12px",
        borderRadius: "9px",
        border: "1px solid #d7d0c1",
        background: "var(--input-bg, #ffffff)",
        color: "var(--text-primary, #2b2925)",
        fontSize: "14px",
        outline: "none",
        transition: "border-color 0.15s, box-shadow 0.15s",
        boxSizing: "border-box",
    },
    inputDirty: {
        borderColor: "#d96c47",
        boxShadow: "0 0 0 3px rgba(217,108,71,0.12)",
    },
    inputError: {
        borderColor: "#e05252",
        boxShadow: "0 0 0 3px rgba(224,82,82,0.12)",
    },
    saveBtn: {
        display: "flex",
        alignItems: "center",
        gap: "5px",
        padding: "9px 14px",
        borderRadius: "9px",
        border: "none",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
    },
    saveBtnActive: {
        background: "#2b2925",
        color: "#fffaf0",
    },
    saveBtnDisabled: {
        background: "#e8e3d9",
        color: "#a09890",
        cursor: "not-allowed",
    },
    toggle: {
        position: "relative",
        width: "44px",
        height: "24px",
        borderRadius: "12px",
        border: "none",
        cursor: "pointer",
        transition: "background 0.2s",
        flexShrink: 0,
    },
    toggleOn: {
        background: "#9a4f35",
    },
    toggleOff: {
        background: "#d7d0c1",
    },
    toggleThumb: {
        position: "absolute",
        top: "3px",
        left: "3px",
        width: "18px",
        height: "18px",
        borderRadius: "50%",
        background: "#fff",
        transition: "transform 0.2s",
        boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
    },
    toggleThumbOn: {
        transform: "translateX(20px)",
    },
    divider: {
        height: "1px",
        background: "var(--card-border, #ede8de)",
        margin: "0 24px",
    },
    skeleton: {
        display: "flex",
        flexDirection: "column",
        gap: "20px",
    },
    skeletonCard: {
        height: "180px",
        borderRadius: "16px",
        background: "linear-gradient(90deg, #f0ece4 25%, #ede8de 50%, #f0ece4 75%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 1.5s infinite",
    },
    toastContainer: {
        position: "fixed",
        bottom: "24px",
        right: "24px",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        zIndex: 9999,
        pointerEvents: "none",
    },
    toast: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "12px 16px",
        borderRadius: "12px",
        fontSize: "13px",
        fontWeight: "500",
        boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
        maxWidth: "360px",
        animation: "slideInRight 0.25s ease",
        pointerEvents: "auto",
    },
    toastSuccess: {
        background: "#2b2925",
        color: "#fffaf0",
        border: "1px solid rgba(255,255,255,0.08)",
    },
    toastError: {
        background: "#a13f24",
        color: "#fff",
        border: "1px solid rgba(255,255,255,0.1)",
    },
};
