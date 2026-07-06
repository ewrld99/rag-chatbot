const FileText = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" /><path d="M14 2v4a2 2 0 0 0 2 2h4" /><path d="M10 9H8" /><path d="M16 13H8" /><path d="M16 17H8" />
    </svg>
);
const Layers = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" />
    </svg>
);
const Database = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5V19a9 3 0 0 0 18 0V5" /><path d="M3 12a9 3 0 0 0 18 0" />
    </svg>
);
const Upload = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
    </svg>
);
const Zap = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
);
const Search = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
);
const MessageSquare = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
);
const CheckCircle = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" />
    </svg>
);
const AlertCircle = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
);

const cards = [
    { key: "documents", label: "Documents", helper: "Uploaded source files", icon: <FileText /> },
    { key: "chunks", label: "Vector Chunks", helper: "Searchable embeddings", icon: <Layers /> },
    { key: "sources", label: "Unique Sources", helper: "Distinct indexed files", icon: <Database /> },
];

const workflowSteps = [
    { icon: <Upload />, text: "Upload PDF, DOCX, or TXT files" },
    { icon: <Zap />, text: "Chunk and embed content automatically" },
    { icon: <Search />, text: "Review indexed document groups" },
    { icon: <MessageSquare />, text: "Serve answers through the public chatbot" },
];

export default function Dashboard({ stats }) {
    return (
        <section style={styles.page}>
            <div style={styles.cardGrid}>
                {cards.map((card) => (
                    <article style={styles.statCard} key={card.key}>
                        <div style={styles.statCardIcon}>{card.icon}</div>
                        <strong style={styles.statValue}>
                            {stats.isLoading ? (
                                <span style={styles.skeleton}>···</span>
                            ) : (
                                stats[card.key] ?? "—"
                            )}
                        </strong>
                        <p style={styles.statLabel}>{card.label}</p>
                        <span style={styles.statHelper}>{card.helper}</span>
                    </article>
                ))}
            </div>

            <div style={styles.panelGrid}>
                <article style={styles.panel}>
                    <div style={styles.panelHeader}>
                        <span style={styles.eyebrow}>Knowledge Workflow</span>
                        <h3 style={styles.panelTitle}>Document Operations</h3>
                    </div>
                    <ul style={styles.workflowList}>
                        {workflowSteps.map((step, i) => (
                            <li key={i} style={styles.workflowItem}>
                                <span style={styles.workflowIcon}>{step.icon}</span>
                                <span style={styles.workflowText}>{step.text}</span>
                            </li>
                        ))}
                    </ul>
                </article>

                <article style={styles.panel}>
                    <div style={styles.panelHeader}>
                        <span style={styles.eyebrow}>Vector Database</span>
                        <h3 style={styles.panelTitle}>
                            {stats.error ? "Connection Issue" : "Index Operational"}
                        </h3>
                    </div>
                    <div style={{ ...styles.statusBadge, ...(stats.error ? styles.statusError : styles.statusOk) }}>
                        {stats.error ? <AlertCircle /> : <CheckCircle />}
                        <span>
                            {stats.error
                                ? stats.error
                                : "The document index is reachable and ready for semantic retrieval."}
                        </span>
                    </div>
                </article>
            </div>
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
    statCard: {
        background: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        borderRadius: "16px",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        position: "relative",
        overflow: "hidden",
        boxShadow: "0 18px 44px rgba(72, 61, 47, 0.1)",
        transition: "border-color 0.2s, transform 0.2s",
    },
    statCardIcon: {
        width: "40px",
        height: "40px",
        borderRadius: "10px",
        background: "var(--app-accent-soft)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--app-accent)",
        marginBottom: "4px",
    },
    statValue: {
        fontSize: "32px",
        fontWeight: "700",
        color: "var(--app-text)",
        letterSpacing: "-0.5px",
        lineHeight: 1,
    },
    skeleton: {
        color: "#c9b8a1",
        animation: "pulse 1.5s ease-in-out infinite",
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
    panelGrid: {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
        gap: "16px",
    },
    panel: {
        background: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        borderRadius: "16px",
        padding: "28px",
        display: "flex",
        flexDirection: "column",
        gap: "20px",
        boxShadow: "0 18px 44px rgba(72, 61, 47, 0.1)",
    },
    panelHeader: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
    },
    eyebrow: {
        fontSize: "11px",
        fontWeight: "700",
        letterSpacing: "1.5px",
        textTransform: "uppercase",
        color: "var(--app-accent)",
    },
    panelTitle: {
        fontSize: "18px",
        fontWeight: "700",
        color: "var(--app-text)",
        margin: 0,
    },
    workflowList: {
        listStyle: "none",
        padding: 0,
        margin: 0,
        display: "flex",
        flexDirection: "column",
        gap: "12px",
    },
    workflowItem: {
        display: "flex",
        alignItems: "center",
        gap: "12px",
        padding: "10px 14px",
        borderRadius: "10px",
        background: "var(--app-panel)",
        border: "1px solid #e3ded2",
    },
    workflowIcon: {
        color: "var(--app-accent)",
        flexShrink: 0,
        display: "flex",
    },
    workflowText: {
        fontSize: "13px",
        color: "#4d4942",
    },
    statusBadge: {
        display: "flex",
        alignItems: "flex-start",
        gap: "10px",
        padding: "14px 16px",
        borderRadius: "10px",
        fontSize: "13px",
        lineHeight: "1.5",
    },
    statusOk: {
        background: "var(--app-accent-soft)",
        border: "1px solid #e8cdb8",
        color: "var(--app-accent)",
    },
    statusError: {
        background: "var(--app-danger-soft)",
        border: "1px solid #f1c4b2",
        color: "var(--app-danger)",
    },
};
