const CheckIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12" />
    </svg>
);

const ClockIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <polyline points="12 6 12 12 16 14" />
    </svg>
);

const StopIcon = () => (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
);

const ActivityIcon = () => (
    <svg className="crawler-spin" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 12a9 9 0 1 1-6.22-8.56" />
    </svg>
);

function formatNumber(value) {
    return new Intl.NumberFormat().format(Number(value || 0));
}

function formatDate(value) {
    if (!value) return "Never";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "Unknown";
    return date.toLocaleString();
}

function statusConfig(status) {
    if (status === "running") {
        return { label: "Running", style: styles.badgeRunning, icon: <ActivityIcon /> };
    }
    if (status === "stopped") {
        return { label: "Stopped", style: styles.badgeStopped, icon: <ClockIcon /> };
    }
    if (status === "finished") {
        return { label: "Finished", style: styles.badgeSuccess, icon: <CheckIcon /> };
    }
    if (status === "error") {
        return { label: "Error", style: styles.badgeError, icon: null };
    }
    if (status === "completed") {
        return { label: "Completed", style: styles.badgeSuccess, icon: <CheckIcon /> };
    }
    return { label: "Idle", style: styles.badgeIdle, icon: <ClockIcon /> };
}

function jobDescription(jobType) {
    if (jobType === "announcements") {
        return "Recent announcements and blog posts, including PDFs found from those selected items.";
    }
    return "Main public website HTML pages. Announcement/blog pages, system portals, PDFs, galleries, and not-found pages are skipped.";
}

function scopeItems(jobType) {
    if (jobType === "announcements") {
        return [
            "10 recent announcements",
            "5 recent blog posts",
            "PDFs from selected items",
            "90-day freshness guard",
        ];
    }
    return [
        "Public HTML pages only",
        "No PDFs",
        "No announcement/blog routes",
        "No portals, staff, gallery, search, or auth pages",
    ];
}

export default function CrawlerDashboard({ title, jobType, data, onCancel }) {
    const status = data?.status || "idle";
    const crawled = Number(data?.crawled || data?.pagesCrawled || 0);
    const max = Number(data?.max || 0);
    const progress = max > 0 ? Math.min(100, Math.round((crawled / max) * 100)) : 0;
    const isRunning = status === "running";
    const badge = statusConfig(status);
    const remaining = Math.max(0, max - crawled);
    const progressTitle = isRunning ? "Progress" : status === "stopped" ? "Saved progress" : "Last progress";

    return (
        <article style={styles.card}>
            <div style={styles.header}>
                <div style={styles.titleBlock}>
                    <div style={styles.titleRow}>
                        <h3 style={styles.title}>{title}</h3>
                        <span style={{ ...styles.badge, ...badge.style }}>
                            {badge.icon}
                            {badge.label}
                        </span>
                    </div>
                    <p style={styles.description}>{jobDescription(jobType)}</p>
                </div>
                {isRunning && onCancel && (
                    <button type="button" onClick={onCancel} style={styles.stopButton}>
                        <StopIcon />
                        Stop
                    </button>
                )}
            </div>

            <div style={styles.body}>
                <div style={styles.progressPanel}>
                    <div style={styles.progressTopline}>
                        <span style={styles.progressLabel}>{progressTitle}</span>
                        <span style={styles.progressValue}>{progress}%</span>
                    </div>
                    <div style={styles.progressTrack}>
                        <div style={{ ...styles.progressFill, width: `${progress}%` }} />
                    </div>
                    <div style={styles.progressMeta}>
                        <span>{formatNumber(crawled)} crawled</span>
                        <span>
                            {max > 0
                                ? `${formatNumber(remaining)} ${isRunning ? "remaining" : "left to resume"}`
                                : "No active limit"}
                        </span>
                    </div>
                </div>

                <div style={styles.metricsGrid}>
                    <div style={styles.metric}>
                        <span style={styles.metricLabel}>Crawled</span>
                        <strong style={styles.metricValue}>{formatNumber(crawled)}</strong>
                    </div>
                    <div style={styles.metric}>
                        <span style={styles.metricLabel}>Limit</span>
                        <strong style={styles.metricValue}>{max > 0 ? formatNumber(max) : "-"}</strong>
                    </div>
                    <div style={styles.metric}>
                        <span style={styles.metricLabel}>Last Run</span>
                        <strong style={styles.metricDate}>{formatDate(data?.last_run || data?.lastRun)}</strong>
                    </div>
                </div>

                <div style={styles.currentPanel}>
                    <span style={styles.currentLabel}>{isRunning ? "Current URL" : "Current URL"}</span>
                    <div style={styles.currentValue}>
                        {data?.current_url || data?.currentItem || (isRunning ? "Starting..." : status === "stopped" ? "Paused. Use Resume Full Crawl to continue." : "No active crawl")}
                    </div>
                </div>

                <div style={styles.scopeGrid}>
                    {scopeItems(jobType).map((item) => (
                        <div key={item} style={styles.scopeItem}>
                            <CheckIcon />
                            <span>{item}</span>
                        </div>
                    ))}
                </div>
            </div>

            <style>{`
                @keyframes crawlerSpin { 100% { transform: rotate(360deg); } }
                .crawler-spin { animation: crawlerSpin 1.4s linear infinite; }
            `}</style>
        </article>
    );
}

const styles = {
    card: {
        background: "var(--card-bg, var(--app-surface))",
        border: "1px solid var(--card-border, var(--app-border))",
        borderRadius: "12px",
        overflow: "hidden",
        boxShadow: "0 4px 16px rgba(72, 61, 47, 0.06)",
    },
    header: {
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "space-between",
        gap: "16px",
        padding: "18px 20px",
        borderBottom: "1px solid var(--card-border, #ede8de)",
    },
    titleBlock: {
        minWidth: 0,
    },
    titleRow: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
        flexWrap: "wrap",
        marginBottom: "5px",
    },
    title: {
        margin: 0,
        fontSize: "16px",
        fontWeight: 700,
        color: "var(--text-primary, var(--app-text))",
    },
    description: {
        margin: 0,
        fontSize: "12px",
        lineHeight: 1.45,
        color: "var(--text-muted, var(--app-faint))",
        maxWidth: "720px",
    },
    badge: {
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        minHeight: "24px",
        padding: "3px 9px",
        borderRadius: "999px",
        fontSize: "12px",
        fontWeight: 700,
        border: "1px solid",
    },
    badgeRunning: {
        background: "#e0f2fe",
        color: "#0369a1",
        borderColor: "#bae6fd",
    },
    badgeSuccess: {
        background: "#dcfce7",
        color: "#15803d",
        borderColor: "#bbf7d0",
    },
    badgeStopped: {
        background: "#fef3c7",
        color: "#92400e",
        borderColor: "#fde68a",
    },
    badgeError: {
        background: "#fee2e2",
        color: "#b91c1c",
        borderColor: "#fecaca",
    },
    badgeIdle: {
        background: "#f3f4f6",
        color: "#4b5563",
        borderColor: "#e5e7eb",
    },
    stopButton: {
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "8px 12px",
        borderRadius: "8px",
        border: "1px solid #fecaca",
        background: "#fee2e2",
        color: "#991b1b",
        fontSize: "13px",
        fontWeight: 700,
        cursor: "pointer",
        whiteSpace: "nowrap",
    },
    body: {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
        gap: "16px",
        padding: "18px 20px",
    },
    progressPanel: {
        gridColumn: "1 / -1",
    },
    progressTopline: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "8px",
    },
    progressLabel: {
        fontSize: "12px",
        fontWeight: 700,
        color: "var(--text-primary, var(--app-text))",
    },
    progressValue: {
        fontSize: "12px",
        fontWeight: 800,
        color: "var(--text-primary, var(--app-text))",
    },
    progressTrack: {
        height: "9px",
        borderRadius: "999px",
        overflow: "hidden",
        background: "var(--app-border, #ede8de)",
    },
    progressFill: {
        height: "100%",
        borderRadius: "999px",
        background: "var(--app-accent, #d96c47)",
        transition: "width 240ms ease",
    },
    progressMeta: {
        display: "flex",
        justifyContent: "space-between",
        gap: "12px",
        marginTop: "7px",
        fontSize: "12px",
        color: "var(--text-muted, var(--app-faint))",
    },
    metricsGrid: {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
        gap: "10px",
    },
    metric: {
        minWidth: 0,
        padding: "12px",
        borderRadius: "8px",
        border: "1px solid var(--app-border)",
        background: "rgba(0,0,0,0.015)",
    },
    metricLabel: {
        display: "block",
        marginBottom: "5px",
        fontSize: "11px",
        fontWeight: 700,
        color: "var(--text-muted, var(--app-faint))",
        textTransform: "uppercase",
    },
    metricValue: {
        display: "block",
        fontSize: "18px",
        color: "var(--text-primary, var(--app-text))",
    },
    metricDate: {
        display: "block",
        fontSize: "12px",
        lineHeight: 1.35,
        color: "var(--text-primary, var(--app-text))",
    },
    currentPanel: {
        minWidth: 0,
        padding: "12px",
        borderRadius: "8px",
        border: "1px solid var(--app-border)",
        background: "var(--input-bg, #fff)",
    },
    currentLabel: {
        display: "block",
        marginBottom: "6px",
        fontSize: "11px",
        fontWeight: 700,
        color: "var(--text-muted, var(--app-faint))",
        textTransform: "uppercase",
    },
    currentValue: {
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        fontSize: "12px",
        lineHeight: 1.45,
        color: "var(--text-primary, var(--app-text))",
        wordBreak: "break-all",
    },
    scopeGrid: {
        gridColumn: "1 / -1",
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: "8px",
    },
    scopeItem: {
        display: "flex",
        alignItems: "center",
        gap: "7px",
        minWidth: 0,
        padding: "9px 10px",
        borderRadius: "8px",
        border: "1px solid var(--app-border)",
        background: "rgba(0,0,0,0.012)",
        color: "var(--text-primary, var(--app-text))",
        fontSize: "12px",
        lineHeight: 1.35,
    },
};
