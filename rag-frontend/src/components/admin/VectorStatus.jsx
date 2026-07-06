import { useEffect, useState } from "react";

const ActivityIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
);
const CpuIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" />
        <line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" />
        <line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" />
        <line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" />
        <line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" />
    </svg>
);
const GridIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" />
    </svg>
);
const FileTextIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
        <path d="M14 2v4a2 2 0 0 0 2 2h4" />
        <path d="M10 9H8" /><path d="M16 13H8" /><path d="M16 17H8" />
    </svg>
);
const CheckCircleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" />
    </svg>
);
const AlertTriangleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
);

const statCards = (stats) => [
    {
        key: "status",
        label: "Status",
        value: stats.error ? "Alert" : "Healthy",
        helper: stats.error || "Vector store is responding",
        icon: <ActivityIcon />,
        accent: stats.error ? "var(--app-danger)" : "var(--app-accent)",
    },
    {
        key: "chunks",
        label: "Embeddings",
        value: stats.isLoading ? "···" : stats.chunks,
        helper: "Total searchable chunks",
        icon: <CpuIcon />,
        accent: "var(--app-accent)",
    },
    {
        key: "documents",
        label: "Coverage",
        value: stats.isLoading ? "···" : stats.documents,
        helper: "Document groups in the index",
        icon: <GridIcon />,
        accent: "var(--app-accent)",
    },
];

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

export default function VectorStatus({ stats, documents }) {
    const isNarrow = useMediaQuery("(max-width: 760px)");

    return (
        <section style={styles.page}>
            {/* Stat Cards */}
            <div style={{ ...styles.cardGrid, ...(isNarrow ? styles.cardGridNarrow : {}) }}>
                {statCards(stats).map((card) => (
                    <article key={card.key} style={{ ...styles.statCard, ...(isNarrow ? styles.statCardNarrow : {}) }}>
                        <div style={{ ...styles.iconBadge, background: `${card.accent}18`, color: card.accent }}>
                            {card.icon}
                        </div>
                        <strong style={{ ...styles.statValue, color: card.accent === "var(--app-accent)" ? "var(--app-text)" : card.accent }}>
                            {card.value ?? "—"}
                        </strong>
                        <p style={styles.statLabel}>{card.label}</p>
                        <span style={styles.statHelper}>{card.helper}</span>
                    </article>
                ))}
            </div>

            {/* Document Table */}
            <article style={{ ...styles.panel, ...(isNarrow ? styles.panelNarrow : {}) }}>
                <div style={{ ...styles.panelHeader, ...(isNarrow ? styles.panelHeaderNarrow : {}) }}>
                    <div>
                        <span style={styles.eyebrow}>Indexed Sources</span>
                        <h3 style={styles.panelTitle}>Vector Database Contents</h3>
                    </div>
                    <span style={styles.countBadge}>
                        {documents.length} {documents.length === 1 ? "source" : "sources"}
                    </span>
                </div>

                <div style={styles.tableWrapper}>
                    {documents.length === 0 && !stats.isLoading ? (
                        <div style={styles.emptyState}>
                            <div style={styles.emptyIcon}><FileTextIcon /></div>
                            <p style={styles.emptyText}>No indexed sources yet. Upload documents to get started.</p>
                        </div>
                    ) : (
                        <div style={styles.table}>
                            <div style={{ ...styles.tableHeader, ...(isNarrow ? styles.tableHeaderNarrow : {}) }}>
                                <span>Source</span>
                                <span>Chunks</span>
                            </div>
                            {documents.map((doc, i) => (
                                <div
                                    key={doc.id}
                                    style={{
                                        ...styles.tableRow,
                                        ...(i % 2 === 0 ? styles.tableRowEven : {}),
                                        ...(isNarrow ? styles.tableRowNarrow : {}),
                                    }}
                                >
                                    <span style={styles.tableRowSource}>
                                        <FileTextIcon />
                                        {doc.source || "Manual entry"}
                                    </span>
                                    <span style={styles.tableRowChunks}>
                                        {doc.chunk_count ?? 1} chunks
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Overall health footer */}
                <div style={{ ...styles.healthBanner, ...(stats.error ? styles.healthBannerError : styles.healthBannerOk) }}>
                    {stats.error ? <AlertTriangleIcon /> : <CheckCircleIcon />}
                    <span style={{ fontSize: "13px" }}>
                        {stats.error ? "Vector store connection issue detected." : "Vector store is healthy and operational."}
                    </span>
                </div>
            </article>
        </section>
    );
}

const styles = {
    page: {
        display: "flex",
        flexDirection: "column",
        gap: "24px",
    },
    cardGrid: {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
        gap: "16px",
    },
    cardGridNarrow: {
        gridTemplateColumns: "1fr",
        gap: "12px",
    },
    statCard: {
        background: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        borderRadius: "16px",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        boxShadow: "0 18px 44px rgba(72, 61, 47, 0.1)",
    },
    statCardNarrow: {
        padding: "18px",
    },
    iconBadge: {
        width: "40px",
        height: "40px",
        borderRadius: "10px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: "6px",
    },
    statValue: {
        fontSize: "30px",
        fontWeight: "700",
        letterSpacing: "-0.5px",
        lineHeight: 1,
    },
    statLabel: {
        fontSize: "14px",
        fontWeight: "600",
        color: "#4d4942",
        margin: 0,
    },
    statHelper: {
        fontSize: "12px",
        color: "var(--app-faint)",
    },
    panel: {
        background: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        borderRadius: "20px",
        padding: "28px",
        display: "flex",
        flexDirection: "column",
        gap: "20px",
        boxShadow: "0 18px 44px rgba(72, 61, 47, 0.1)",
    },
    panelNarrow: {
        padding: "16px",
        borderRadius: "14px",
    },
    panelHeader: {
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "space-between",
        gap: "16px",
    },
    panelHeaderNarrow: {
        flexDirection: "column",
        gap: "10px",
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
    panelTitle: {
        fontSize: "18px",
        fontWeight: "700",
        color: "var(--app-text)",
        margin: 0,
    },
    countBadge: {
        padding: "4px 12px",
        borderRadius: "20px",
        fontSize: "12px",
        fontWeight: "600",
        background: "var(--app-accent-soft)",
        border: "1px solid #e8cdb8",
        color: "var(--app-accent)",
        whiteSpace: "nowrap",
    },
    tableWrapper: {
        borderRadius: "12px",
        overflow: "hidden",
        border: "1px solid #e3ded2",
    },
    table: {
        display: "flex",
        flexDirection: "column",
    },
    tableHeader: {
        display: "flex",
        justifyContent: "space-between",
        padding: "10px 16px",
        background: "var(--app-panel)",
        fontSize: "11px",
        fontWeight: "700",
        letterSpacing: "1px",
        textTransform: "uppercase",
        color: "var(--app-faint)",
    },
    tableHeaderNarrow: {
        display: "none",
    },
    tableRow: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "12px 16px",
        borderTop: "1px solid #e3ded2",
        transition: "background 0.15s",
    },
    tableRowNarrow: {
        alignItems: "flex-start",
        flexDirection: "column",
        gap: "8px",
        padding: "14px",
    },
    tableRowEven: {
        background: "var(--app-surface-muted)",
    },
    tableRowSource: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        fontSize: "13px",
        color: "#4d4942",
        minWidth: 0,
        overflowWrap: "anywhere",
    },
    tableRowChunks: {
        fontSize: "12px",
        fontWeight: "600",
        color: "var(--app-accent)",
        background: "var(--app-accent-soft)",
        padding: "3px 10px",
        borderRadius: "20px",
    },
    emptyState: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "10px",
        padding: "40px 20px",
        textAlign: "center",
    },
    emptyIcon: {
        width: "44px",
        height: "44px",
        borderRadius: "12px",
        background: "var(--app-bg)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--app-faint)",
    },
    emptyText: {
        fontSize: "13px",
        color: "var(--app-faint)",
        margin: 0,
    },
    healthBanner: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "12px 16px",
        borderRadius: "10px",
    },
    healthBannerOk: {
        background: "var(--app-accent-soft)",
        border: "1px solid #e8cdb8",
        color: "var(--app-accent)",
    },
    healthBannerError: {
        background: "var(--app-danger-soft)",
        border: "1px solid #f1c4b2",
        color: "var(--app-danger)",
    },
};
