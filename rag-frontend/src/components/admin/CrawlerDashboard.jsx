import React, { useEffect, useRef } from "react";

// --- Icons ---
const CheckIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
);
const SpinnerIcon = () => (
    <svg className="spin-anim" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="4.93" x2="19.07" y2="7.76"/></svg>
);
const WarningIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
);
const ErrorIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
);
const ClockIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
);
const DownloadIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
);

const ALL_STAGES = [
    "Connect to Website",
    "Discover Pages",
    "Crawling Website",
    "Downloading Documents",
    "Extracting Text",
    "Generating Embeddings",
    "Saving Knowledge Base"
];

function formatNumber(num) {
    if (num === undefined || num === null) return "0";
    return new Intl.NumberFormat().format(num);
}

// ─── Completion Screen ─────────────────────────────────────────────────────────
function CompletionScreen({ data, title, onCancel }) {
    return (
        <div style={styles.card}>
            <div style={{ ...styles.cardHeader, display: 'flex', justifyContent: 'space-between' }}>
                <div>
                    <h3 style={styles.cardTitle}>{title}</h3>
                    <p style={styles.cardDescription}>Knowledge Base Created Successfully</p>
                </div>
                <div style={styles.badgeSuccess}>Completed</div>
            </div>
            <div style={styles.cardBody}>
                <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                    <div style={styles.summaryCard}>
                        <div style={styles.statLabel}>Pages Crawled</div>
                        <div style={styles.statValue}>{formatNumber(data.pagesCrawled)}</div>
                    </div>
                    <div style={styles.summaryCard}>
                        <div style={styles.statLabel}>Files Downloaded</div>
                        <div style={styles.statValue}>{formatNumber(data.filesDownloaded)}</div>
                    </div>
                    <div style={styles.summaryCard}>
                        <div style={styles.statLabel}>Documents Indexed</div>
                        <div style={styles.statValue}>{formatNumber(data.documentsProcessed)}</div>
                    </div>
                    <div style={styles.summaryCard}>
                        <div style={styles.statLabel}>Chunks Created</div>
                        <div style={styles.statValue}>{formatNumber(data.chunksCreated)}</div>
                    </div>
                    <div style={styles.summaryCard}>
                        <div style={styles.statLabel}>Embeddings Generated</div>
                        <div style={styles.statValue}>{formatNumber(data.embeddingsGenerated)}</div>
                    </div>
                    <div style={styles.summaryCard}>
                        <div style={styles.statLabel}>Duration</div>
                        <div style={{...styles.statValue, fontSize: "16px"}}>{data.elapsed || "00:00:00"}</div>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '24px', marginTop: '24px', flexWrap: 'wrap' }}>
                    {(data.warnings?.length > 0 || data.errors?.length > 0) && (
                        <div style={{ flex: 1, minWidth: '250px' }}>
                            <h4 style={styles.subTitle}>Issues & Warnings</h4>
                            <ul style={styles.issueList}>
                                {data.errors?.map((err, i) => <li key={i}><ErrorIcon /> {err}</li>)}
                                {data.warnings?.map((warn, i) => <li key={i}><WarningIcon /> {warn}</li>)}
                            </ul>
                        </div>
                    )}
                    {data.fileBreakdown && (
                        <div style={{ flex: 1, minWidth: '250px' }}>
                            <h4 style={styles.subTitle}>File Types Indexed</h4>
                            <div style={styles.fileBreakdownGrid}>
                                {Object.entries(data.fileBreakdown).map(([ext, count]) => (
                                    <div key={ext} style={styles.fileBreakdownItem}>
                                        <span style={styles.fileExt}>{ext}</span>
                                        <span style={styles.fileCount}>{formatNumber(count)}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
            <div style={styles.cardFooter}>
                <button onClick={onCancel} style={styles.actionBtn}>Start New Crawl</button>
            </div>
        </div>
    );
}

// ─── Main Dashboard ────────────────────────────────────────────────────────────
export default function CrawlerDashboard({ title, data, onCancel }) {
    const feedContainerRef = useRef(null);

    // Auto-scroll activity feed
    useEffect(() => {
        if (feedContainerRef.current) {
            feedContainerRef.current.scrollTop = feedContainerRef.current.scrollHeight;
        }
    }, [data?.recentEvents]);

    if (!data || data.status === "idle") {
        return (
            <div style={styles.card}>
                <div style={styles.cardHeader}>
                    <h3 style={styles.cardTitle}>{title}</h3>
                    <p style={styles.cardDescription}>Ready to start crawling.</p>
                </div>
            </div>
        );
    }

    if (data.status === "completed") {
        return <CompletionScreen title={title} data={data} onCancel={onCancel} />;
    }

    const stageIndex = Math.max(0, ALL_STAGES.findIndex(s => s.toLowerCase() === (data.stage || "").toLowerCase()));
    const isError = data.status === "error";
    const isPaused = data.status === "paused";
    const progressColor = isError ? "#ef4444" : isPaused ? "#f59e0b" : "#3b82f6";
    const badgeStyle = isError ? styles.badgeError : isPaused ? styles.badgeWarning : styles.badgeRunning;

    return (
        <div style={styles.card}>
            {/* Header & Overall Progress */}
            <div style={styles.cardHeader}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                    <div>
                        <h3 style={styles.cardTitle}>{title}</h3>
                        <p style={styles.cardDescription}>Data Ingestion Pipeline</p>
                    </div>
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                        <div style={badgeStyle}>
                            {isError ? "Error" : isPaused ? "Paused" : "Running"}
                        </div>
                        {onCancel && (
                            <button onClick={onCancel} style={styles.stopBtn}>Stop / Reset</button>
                        )}
                    </div>
                </div>

                <div style={styles.progressContainer}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span style={styles.progressLabel}>Overall Progress</span>
                        <span style={styles.progressPercent}>{data.overallProgress || 0}%</span>
                    </div>
                    <div style={styles.progressBarBg}>
                        <div 
                            style={{ 
                                ...styles.progressBarFill, 
                                width: `${Math.min(100, Math.max(0, data.overallProgress || 0))}%`,
                                backgroundColor: progressColor
                            }} 
                        />
                    </div>
                </div>
            </div>

            <div style={styles.cardBody}>
                <div style={styles.gridTwoCol}>
                    {/* Left Col: Stages & Current Activity */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                        
                        <div>
                            <h4 style={styles.subTitle}>Pipeline Stages</h4>
                            <div style={styles.stageList}>
                                {ALL_STAGES.map((stage, idx) => {
                                    const isCompleted = idx < stageIndex;
                                    const isCurrent = idx === stageIndex && data.status === "running";
                                    const isPending = idx > stageIndex || (idx === stageIndex && data.status !== "running");

                                    return (
                                        <div key={stage} style={{ ...styles.stageItem, opacity: isPending ? 0.5 : 1 }}>
                                            <div style={styles.stageIconBox}>
                                                {isCompleted ? <CheckIcon /> : isCurrent ? <SpinnerIcon /> : <div style={styles.circleEmpty} />}
                                            </div>
                                            <div style={{ flex: 1 }}>
                                                <div style={{ ...styles.stageName, fontWeight: isCurrent ? '600' : '500' }}>
                                                    {stage}
                                                </div>
                                                {isCurrent && (
                                                    <div style={styles.stageSubProgressBg}>
                                                        <div style={{ 
                                                            ...styles.stageSubProgressFill, 
                                                            width: `${data.stageProgress || 0}%` 
                                                        }} />
                                                    </div>
                                                )}
                                            </div>
                                            {isCurrent && <div style={styles.stagePercent}>{data.stageProgress || 0}%</div>}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        <div style={styles.activityBox}>
                            <h4 style={styles.subTitle}>Current Activity</h4>
                            <div style={styles.activityItem}>
                                <div style={styles.activityLabel}>{data.stage || "Working"}:</div>
                                <div style={styles.activityValue}>{data.currentItem || "..."}</div>
                            </div>
                        </div>

                    </div>

                    {/* Right Col: Stats, Feed, Performance */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                        
                        <div>
                            <h4 style={styles.subTitle}>Live Statistics</h4>
                            <div style={styles.statsGrid}>
                                <div style={styles.statBox}>
                                    <div style={styles.statLabel}>Discovered</div>
                                    <div style={styles.statValue}>{formatNumber(data.pagesDiscovered)}</div>
                                </div>
                                <div style={styles.statBox}>
                                    <div style={styles.statLabel}>Crawled</div>
                                    <div style={styles.statValue}>{formatNumber(data.pagesCrawled)}</div>
                                </div>
                                <div style={styles.statBox}>
                                    <div style={styles.statLabel}>Downloaded</div>
                                    <div style={styles.statValue}>{formatNumber(data.filesDownloaded)}</div>
                                </div>
                                <div style={styles.statBox}>
                                    <div style={styles.statLabel}>Processed</div>
                                    <div style={styles.statValue}>{formatNumber(data.documentsProcessed)}</div>
                                </div>
                                <div style={styles.statBox}>
                                    <div style={styles.statLabel}>Chunks</div>
                                    <div style={styles.statValue}>{formatNumber(data.chunksCreated)}</div>
                                </div>
                                <div style={styles.statBox}>
                                    <div style={styles.statLabel}>Inserted</div>
                                    <div style={styles.statValue}>{formatNumber(data.databaseInserted)}</div>
                                </div>
                            </div>
                        </div>

                        <div>
                            <h4 style={styles.subTitle}>Performance Metrics</h4>
                            <div style={styles.perfGrid}>
                                <div style={styles.perfItem}>
                                    <ClockIcon /> <span>Elapsed: {data.elapsed || "00:00:00"}</span>
                                </div>
                                <div style={styles.perfItem}>
                                    <ClockIcon /> <span style={{color: 'var(--app-faint)'}}>Remaining: {data.remaining || "Calculating..."}</span>
                                </div>
                                <div style={styles.perfItem}>
                                    <DownloadIcon /> <span>{data.downloadSpeed || "0 MB/s"}</span>
                                </div>
                                <div style={styles.perfItem}>
                                    <DownloadIcon /> <span>{data.processingSpeed || "0 pages/min"}</span>
                                </div>
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: '16px', flex: 1, minHeight: '200px' }}>
                            <div style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
                                <h4 style={styles.subTitle}>Activity Feed</h4>
                                <div style={styles.feedBox} ref={feedContainerRef}>
                                    {(data.recentEvents || []).map((event, i) => (
                                        <div key={i} style={styles.feedItem}>
                                            <span style={styles.feedTime}>{event.time}</span>
                                            {event.type === 'success' ? <span style={{color: '#10b981'}}><CheckIcon/></span> : 
                                             event.type === 'warning' ? <WarningIcon/> : 
                                             <div style={{width: '16px', height: '16px', borderRadius: '50%', background: '#3b82f6', transform: 'scale(0.5)'}}/>}
                                            <span style={styles.feedText}>{event.text}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                                <h4 style={styles.subTitle}>Warnings</h4>
                                <div style={styles.feedBox}>
                                    {(data.warnings || []).length === 0 ? (
                                        <div style={{ color: 'var(--app-faint)', fontSize: '12px', padding: '8px' }}>No warnings</div>
                                    ) : (
                                        data.warnings.map((warn, i) => (
                                            <div key={i} style={styles.feedItem}>
                                                <WarningIcon />
                                                <span style={{...styles.feedText, color: '#f59e0b'}}>{warn}</span>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            {/* Injected CSS for spin animation */}
            <style>{`
                @keyframes spin { 100% { transform: rotate(360deg); } }
                .spin-anim { animation: spin 2s linear infinite; color: #3b82f6; }
            `}</style>
        </div>
    );
}

// ─── Styles ────────────────────────────────────────────────────────────────────
const styles = {
    card: {
        background: "var(--card-bg, var(--app-surface))",
        border: "1px solid var(--card-border, var(--app-border))",
        borderRadius: "16px",
        overflow: "hidden",
        boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
        fontFamily: "system-ui, -apple-system, sans-serif",
        marginBottom: "24px",
    },
    cardHeader: {
        padding: "20px 24px",
        borderBottom: "1px solid var(--card-border, #ede8de)",
        background: "rgba(0,0,0,0.01)",
    },
    cardBody: {
        padding: "24px",
    },
    cardFooter: {
        padding: "16px 24px",
        borderTop: "1px solid var(--card-border, #ede8de)",
        background: "rgba(0,0,0,0.01)",
        display: "flex",
        justifyContent: "flex-end",
    },
    cardTitle: {
        fontSize: "18px",
        fontWeight: "700",
        color: "var(--text-primary, var(--app-text))",
        margin: "0 0 4px",
    },
    cardDescription: {
        fontSize: "13px",
        color: "var(--text-muted, var(--app-faint))",
        margin: 0,
    },
    badgeRunning: {
        padding: "4px 10px", borderRadius: "12px", fontSize: "12px", fontWeight: "600",
        background: "#e0f2fe", color: "#0284c7", border: "1px solid #bae6fd",
    },
    badgeWarning: {
        padding: "4px 10px", borderRadius: "12px", fontSize: "12px", fontWeight: "600",
        background: "#fef3c7", color: "#d97706", border: "1px solid #fde68a",
    },
    badgeError: {
        padding: "4px 10px", borderRadius: "12px", fontSize: "12px", fontWeight: "600",
        background: "#fee2e2", color: "#dc2626", border: "1px solid #fecaca",
    },
    badgeSuccess: {
        padding: "4px 10px", borderRadius: "12px", fontSize: "12px", fontWeight: "600",
        background: "#d1fae5", color: "#059669", border: "1px solid #a7f3d0", height: "fit-content"
    },
    stopBtn: {
        padding: "6px 12px", borderRadius: "8px", fontSize: "12px", fontWeight: "600",
        background: "#ef4444", color: "white", border: "none", cursor: "pointer",
    },
    actionBtn: {
        padding: "8px 16px", borderRadius: "8px", fontSize: "14px", fontWeight: "600",
        background: "var(--app-text)", color: "var(--app-surface)", border: "none", cursor: "pointer",
    },
    progressContainer: {
        marginTop: "16px",
    },
    progressLabel: {
        fontSize: "13px", fontWeight: "600", color: "var(--text-primary, var(--app-text))",
    },
    progressPercent: {
        fontSize: "13px", fontWeight: "700", color: "var(--text-primary, var(--app-text))",
    },
    progressBarBg: {
        width: "100%", height: "10px", background: "var(--app-border, #ede8de)", borderRadius: "5px", overflow: "hidden",
    },
    progressBarFill: {
        height: "100%", transition: "width 0.4s ease-in-out, background-color 0.4s",
    },
    gridTwoCol: {
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "32px",
    },
    subTitle: {
        fontSize: "14px", fontWeight: "600", margin: "0 0 12px", color: "var(--text-primary, var(--app-text))",
        textTransform: "uppercase", letterSpacing: "0.5px",
    },
    stageList: {
        display: "flex", flexDirection: "column", gap: "12px",
    },
    stageItem: {
        display: "flex", alignItems: "center", gap: "12px", transition: "opacity 0.3s",
    },
    stageIconBox: {
        width: "24px", height: "24px", display: "flex", alignItems: "center", justifyContent: "center",
        color: "#10b981",
    },
    circleEmpty: {
        width: "12px", height: "12px", borderRadius: "50%", border: "2px solid var(--app-border-strong)",
    },
    stageName: {
        fontSize: "13px", color: "var(--text-primary, var(--app-text))",
    },
    stageSubProgressBg: {
        width: "100%", height: "4px", background: "var(--app-border)", borderRadius: "2px", marginTop: "6px", overflow: "hidden",
    },
    stageSubProgressFill: {
        height: "100%", background: "#3b82f6", transition: "width 0.3s",
    },
    stagePercent: {
        fontSize: "12px", fontWeight: "600", color: "var(--app-faint)", minWidth: "32px", textAlign: "right",
    },
    activityBox: {
        background: "rgba(0,0,0,0.02)", border: "1px solid var(--app-border)", borderRadius: "12px", padding: "16px",
    },
    activityItem: {
        display: "flex", flexDirection: "column", gap: "4px",
    },
    activityLabel: {
        fontSize: "12px", color: "var(--app-faint)", fontWeight: "600",
    },
    activityValue: {
        fontSize: "13px", color: "var(--text-primary)", wordBreak: "break-all", fontFamily: "monospace",
    },
    statsGrid: {
        display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px",
    },
    statBox: {
        background: "rgba(0,0,0,0.015)", border: "1px solid var(--app-border)", borderRadius: "10px", padding: "12px",
    },
    statLabel: {
        fontSize: "11px", color: "var(--app-faint)", textTransform: "uppercase", fontWeight: "600", marginBottom: "4px",
    },
    statValue: {
        fontSize: "18px", fontWeight: "700", color: "var(--text-primary)",
    },
    perfGrid: {
        display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px",
    },
    perfItem: {
        display: "flex", alignItems: "center", gap: "8px", fontSize: "12px", color: "var(--text-primary)",
        background: "rgba(0,0,0,0.015)", border: "1px solid var(--app-border)", borderRadius: "8px", padding: "8px 12px",
    },
    feedBox: {
        flex: 1, border: "1px solid var(--app-border)", borderRadius: "8px", padding: "12px",
        background: "var(--input-bg, #fff)", overflowY: "auto", maxHeight: "200px",
        display: "flex", flexDirection: "column", gap: "8px",
    },
    feedItem: {
        display: "flex", alignItems: "flex-start", gap: "8px", fontSize: "12px",
    },
    feedTime: {
        color: "var(--app-faint)", fontSize: "11px", minWidth: "50px",
    },
    feedText: {
        color: "var(--text-primary)", wordBreak: "break-word",
    },
    summaryCard: {
        flex: "1 1 150px", background: "var(--app-surface)", border: "1px solid var(--app-border)", 
        borderRadius: "12px", padding: "16px", textAlign: "center",
        boxShadow: "0 2px 8px rgba(0,0,0,0.02)",
    },
    issueList: {
        margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: "8px",
        fontSize: "13px", color: "var(--text-primary)",
    },
    fileBreakdownGrid: {
        display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px",
    },
    fileBreakdownItem: {
        display: "flex", justifyContent: "space-between", padding: "8px 12px", 
        background: "rgba(0,0,0,0.02)", borderRadius: "6px", fontSize: "13px",
    },
    fileExt: {
        fontWeight: "600",
    },
    fileCount: {
        color: "var(--app-faint)",
    }
};
