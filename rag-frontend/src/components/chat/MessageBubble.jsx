import { memo } from "react";
import udomLogo from "../../assets/udom-logo.svg";

const BotLogo = () => (
    <img src={udomLogo} alt="The University of Dodoma" style={styles.logoImage} />
);

const UserIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
    </svg>
);

const ThumbsUpIcon = ({ filled }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
    </svg>
);

const ThumbsDownIcon = ({ filled }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
    </svg>
);

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function safeSourceUrl(url) {
    if (!url) return null;
    try {
        const parsed = new URL(url, window.location.origin);
        return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : null;
    } catch {
        return null;
    }
}

function SourceList({ sources }) {
    if (!Array.isArray(sources) || sources.length === 0) return null;
    return (
        <section style={styles.sources} aria-label="Sources">
            <span style={styles.sourcesTitle}>Sources</span>
            <ul style={styles.sourcesList}>
                {sources.map((source, index) => {
                    const href = safeSourceUrl(source.url);
                    const label = source.name || "UDOM document";
                    return (
                        <li key={source.document_id || `${label}-${index}`} style={styles.sourceItem}>
                            {href
                                ? <a href={href} target="_blank" rel="noopener noreferrer" style={styles.sourceLink}>{label}</a>
                                : <span>{label}</span>}
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}

const MODEL_LABELS = {
    "llama3.2:3b": "Llama 3.2 3B Local",
    "gemma3:4b": "Gemma 3 4B Local",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant",
};

const MessageBubble = memo(function MessageBubble({
    id,
    role,
    content,
    sources = [],
    feedback,
    onFeedback,
    selectedModel,
    fallbackUsed = false,
}) {
    const isUser = role === "user";

    const formattedContent = typeof content === "string"
        ? content.replace(/([^\n])\n([^\n\s\-*\d#|>])/g, "$1\n\n$2")
        : content;

    return (
        <article style={{ ...styles.row, ...(isUser ? styles.rowUser : styles.rowAssistant) }}>
            {!isUser && (
                <div style={styles.avatar}>
                    <BotLogo />
                </div>
            )}

            <div style={{ ...styles.content, ...(isUser ? styles.contentUser : styles.contentAssistant) }}>
                {!isUser && (
                    <div style={styles.authorRow}>
                        <span style={styles.author}>Assistant</span>
                        {selectedModel && (
                            <span
                                style={styles.modelNote}
                                title={fallbackUsed ? "The selected model was unavailable, so a fallback answered." : "Model used for this answer"}
                            >
                                {MODEL_LABELS[selectedModel] || selectedModel}
                                {fallbackUsed ? " (fallback)" : ""}
                            </span>
                        )}
                    </div>
                )}
                <div style={{ ...styles.bubble, ...(isUser ? styles.bubbleUser : styles.bubbleAssistant), whiteSpace: isUser ? "pre-wrap" : "normal" }}>
                    {isUser ? content : (
                        <ReactMarkdown 
                            remarkPlugins={[remarkGfm]}
                            components={{
                                a: ({children}) => <span>{children}</span>,
                                p: ({node, ...props}) => <p style={{ margin: "0 0 12px 0", lineHeight: "1.6", ...(node.parent && node.parent.tagName === 'li' ? { margin: 0 } : {}) }} {...props} />,
                                h1: (props) => <h1 style={{ margin: "16px 0 8px 0", fontSize: "1.25em", fontWeight: 600 }} {...props} />,
                                h2: (props) => <h2 style={{ margin: "16px 0 8px 0", fontSize: "1.15em", fontWeight: 600 }} {...props} />,
                                h3: (props) => <h3 style={{ margin: "14px 0 6px 0", fontSize: "1.05em", fontWeight: 600 }} {...props} />,
                                ul: (props) => <ul style={{ margin: "0 0 12px 0", paddingLeft: "20px" }} {...props} />,
                                ol: (props) => <ol style={{ margin: "0 0 12px 0", paddingLeft: "20px" }} {...props} />,
                                li: (props) => <li style={{ margin: "4px 0", lineHeight: "1.5" }} {...props} />,
                                table: (props) => <table style={{ borderCollapse: "collapse", width: "100%", marginBottom: "16px" }} {...props} />,
                                th: (props) => <th style={{ border: "1px solid var(--app-border)", padding: "8px", backgroundColor: "var(--app-bg)" }} {...props} />,
                                td: (props) => <td style={{ border: "1px solid var(--app-border)", padding: "8px" }} {...props} />
                            }}
                        >
                            {formattedContent}
                        </ReactMarkdown>
                    )}
                </div>
                {!isUser && <SourceList sources={sources} />}
                {!isUser && id && (
                    <div style={styles.feedbackActions}>
                        <button 
                            type="button" 
                            style={{ ...styles.feedbackBtn, ...(feedback === 1 ? styles.feedbackBtnActive : {}) }}
                            onClick={() => onFeedback && onFeedback(id, feedback === 1 ? 0 : 1)}
                            aria-label="Thumbs up"
                            title="Helpful"
                        >
                            <ThumbsUpIcon filled={feedback === 1} />
                        </button>
                        <button 
                            type="button" 
                            style={{ ...styles.feedbackBtn, ...(feedback === -1 ? styles.feedbackBtnActive : {}) }}
                            onClick={() => onFeedback && onFeedback(id, feedback === -1 ? 0 : -1)}
                            aria-label="Thumbs down"
                            title="Not helpful"
                        >
                            <ThumbsDownIcon filled={feedback === -1} />
                        </button>
                    </div>
                )}
            </div>

            {isUser && (
                <div style={styles.avatarUser}>
                    <UserIcon />
                </div>
            )}
        </article>
    );
});

export default MessageBubble;

export function TypingBubble({ status = "" }) {
    return (
        <article style={styles.row}>
            <div style={styles.avatar}>
                <BotLogo />
            </div>
            <div style={styles.content}>
                <span style={styles.author}>Assistant</span>
                {status && <span style={styles.typingStatus}>{status}</span>}
                <div style={{ ...styles.bubble, ...styles.bubbleAssistant, ...styles.typingBubble }} aria-label="Assistant is thinking">
                    <span style={{ ...styles.dot, animationDelay: "0ms" }} />
                    <span style={{ ...styles.dot, animationDelay: "180ms" }} />
                    <span style={{ ...styles.dot, animationDelay: "360ms" }} />
                </div>
            </div>
        </article>
    );
}

const styles = {
    row: {
        width: "min(780px, 100%)",
        display: "flex",
        alignItems: "flex-start",
        gap: "12px",
        padding: "14px 0",
        margin: "0 auto",
    },
    rowUser: {
        flexDirection: "row-reverse",
        justifyContent: "flex-start",
    },
    rowAssistant: {
        flexDirection: "row",
    },
    avatar: {
        width: "32px",
        height: "32px",
        borderRadius: "50%",
        background: "var(--app-surface-muted)",
        border: "1px solid var(--app-border-strong)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        marginTop: "2px",
        overflow: "hidden",
    },
    logoImage: {
        width: "100%",
        height: "100%",
        objectFit: "cover",
        display: "block",
    },
    avatarUser: {
        width: "32px",
        height: "32px",
        borderRadius: "10px",
        background: "var(--app-surface-muted)",
        border: "1px solid var(--app-border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--app-muted)",
        flexShrink: 0,
        marginTop: "2px",
    },
    content: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        maxWidth: "min(680px, 84%)",
    },
    contentUser: {
        alignItems: "flex-end",
    },
    contentAssistant: {
        alignItems: "flex-start",
    },
    author: {
        fontSize: "12px",
        fontWeight: "700",
        color: "var(--app-muted)",
        letterSpacing: 0,
        textTransform: "none",
        paddingLeft: "2px",
    },
    authorRow: {
        display: "flex",
        alignItems: "center",
        flexWrap: "wrap",
        gap: "7px",
        minHeight: "18px",
    },
    modelNote: {
        fontSize: "11px",
        color: "var(--app-faint)",
        overflowWrap: "anywhere",
    },
    bubble: {
        padding: "0",
        borderRadius: "16px",
        fontSize: "15px",
        lineHeight: "1.7",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
    },
    bubbleUser: {
        padding: "10px 15px",
        background: "var(--app-surface-muted)",
        color: "var(--app-text)",
        fontWeight: "400",
        borderBottomRightRadius: "6px",
    },
    bubbleAssistant: {
        background: "transparent",
        border: "none",
        color: "var(--app-text)",
        borderBottomLeftRadius: "16px",
    },
    sources: {
        width: "100%",
        paddingTop: "10px",
        marginTop: "2px",
        borderTop: "1px solid var(--app-border)",
        color: "var(--app-muted)",
    },
    sourcesTitle: {
        display: "block",
        marginBottom: "5px",
        fontSize: "12px",
        fontWeight: "700",
        letterSpacing: 0,
    },
    sourcesList: {
        margin: 0,
        paddingLeft: "18px",
        display: "grid",
        gap: "3px",
    },
    sourceItem: {
        fontSize: "12px",
        lineHeight: "1.45",
        overflowWrap: "anywhere",
    },
    sourceLink: {
        color: "var(--app-accent)",
        fontWeight: "600",
        textDecoration: "underline",
        textUnderlineOffset: "2px",
    },
    typingBubble: {
        display: "flex",
        alignItems: "center",
        gap: "5px",
        padding: "10px 0",
    },
    typingStatus: {
        display: "block",
        margin: "3px 0 6px",
        color: "var(--app-muted)",
        fontSize: "12px",
        fontWeight: "600",
    },
    dot: {
        width: "7px",
        height: "7px",
        borderRadius: "50%",
        background: "var(--app-accent)",
        display: "inline-block",
        animation: "typingPulse 1.2s ease-in-out infinite",
    },
    feedbackActions: {
        display: "flex",
        gap: "6px",
        marginTop: "2px",
        paddingLeft: "2px",
    },
    feedbackBtn: {
        background: "transparent",
        border: "none",
        color: "var(--app-muted)",
        cursor: "pointer",
        padding: "4px",
        borderRadius: "4px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transition: "color 0.2s, background 0.2s",
    },
    feedbackBtnActive: {
        color: "var(--app-accent)",
    },
};
