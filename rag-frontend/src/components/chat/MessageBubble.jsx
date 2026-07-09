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

const renderMarkdown = (text) => {
    if (typeof text !== 'string') return text;
    
    // Step 1: Parse links
    const linkRegex = /\[(.*?)\]\((.*?)\)/g;
    const partsWithLinks = [];
    let lastIndex = 0;
    let match;

    while ((match = linkRegex.exec(text)) !== null) {
        if (match.index > lastIndex) {
            partsWithLinks.push(text.substring(lastIndex, match.index));
        }
        partsWithLinks.push(
            <a key={`link-${match.index}`} href={match[2]} target="_blank" rel="noopener noreferrer" style={{ color: "#63b3a4", textDecoration: "underline", fontWeight: "600" }}>
                {match[1]}
            </a>
        );
        lastIndex = linkRegex.lastIndex;
    }
    if (lastIndex < text.length) {
        partsWithLinks.push(text.substring(lastIndex));
    }

    // Step 2: Parse bold within string chunks
    const finalParts = [];
    const boldRegex = /\*\*(.*?)\*\*/g;

    partsWithLinks.forEach((part, i) => {
        if (typeof part !== 'string') {
            finalParts.push(part);
            return;
        }

        let lastBIndex = 0;
        let bMatch;
        while ((bMatch = boldRegex.exec(part)) !== null) {
            if (bMatch.index > lastBIndex) {
                finalParts.push(part.substring(lastBIndex, bMatch.index));
            }
            finalParts.push(
                <strong key={`bold-${i}-${bMatch.index}`} style={{ fontWeight: "700" }}>
                    {bMatch[1]}
                </strong>
            );
            lastBIndex = boldRegex.lastIndex;
        }
        if (lastBIndex < part.length) {
            finalParts.push(part.substring(lastBIndex));
        }
    });

    return finalParts;
};

const MessageBubble = memo(function MessageBubble({ id, role, content, feedback, onFeedback }) {
    const isUser = role === "user";

    return (
        <article style={{ ...styles.row, ...(isUser ? styles.rowUser : styles.rowAssistant) }}>
            {!isUser && (
                <div style={styles.avatar}>
                    <BotLogo />
                </div>
            )}

            <div style={{ ...styles.content, ...(isUser ? styles.contentUser : styles.contentAssistant) }}>
                {!isUser && <span style={styles.author}>Assistant</span>}
                <div style={{ ...styles.bubble, ...(isUser ? styles.bubbleUser : styles.bubbleAssistant), whiteSpace: "pre-wrap" }}>
                    {isUser ? content : renderMarkdown(content)}
                </div>
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

export function TypingBubble() {
    return (
        <article style={styles.row}>
            <div style={styles.avatar}>
                <BotLogo />
            </div>
            <div style={styles.content}>
                <span style={styles.author}>Assistant</span>
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
        display: "flex",
        alignItems: "flex-end",
        gap: "10px",
        padding: "6px 0",
    },
    rowUser: {
        flexDirection: "row-reverse",
    },
    rowAssistant: {
        flexDirection: "row",
    },
    avatar: {
        width: "32px",
        height: "32px",
        borderRadius: "10px",
        background: "linear-gradient(135deg, rgba(99,179,164,0.15), rgba(99,179,164,0.05))",
        border: "1px solid rgba(99,179,164,0.2)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#63b3a4",
        flexShrink: 0,
    },
    logoImage: {
        width: "100%",
        height: "100%",
        objectFit: "cover",
        borderRadius: "50%",
        display: "block",
    },
    avatarUser: {
        width: "32px",
        height: "32px",
        borderRadius: "10px",
        background: "rgba(255,255,255,0.05)",
        border: "1px solid rgba(255,255,255,0.09)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#718096",
        flexShrink: 0,
    },
    content: {
        display: "flex",
        flexDirection: "column",
        gap: "4px",
        maxWidth: "72%",
    },
    contentUser: {
        alignItems: "flex-end",
    },
    contentAssistant: {
        alignItems: "flex-start",
    },
    author: {
        fontSize: "11px",
        fontWeight: "600",
        color: "#63b3a4",
        letterSpacing: "0.5px",
        textTransform: "uppercase",
        paddingLeft: "2px",
    },
    bubble: {
        padding: "11px 15px",
        borderRadius: "14px",
        fontSize: "14px",
        lineHeight: "1.6",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
    },
    bubbleUser: {
        background: "linear-gradient(135deg, #63b3a4, #4a9080)",
        color: "#0d1520",
        fontWeight: "400",
        borderBottomRightRadius: "4px",
    },
    bubbleAssistant: {
        background: "#1e2d40",
        border: "1px solid rgba(255,255,255,0.07)",
        color: "#e2e8f0",
        borderBottomLeftRadius: "4px",
    },
    typingBubble: {
        display: "flex",
        alignItems: "center",
        gap: "5px",
        padding: "14px 18px",
    },
    dot: {
        width: "7px",
        height: "7px",
        borderRadius: "50%",
        background: "#4a9080",
        display: "inline-block",
        animation: "typingBounce 1.2s ease-in-out infinite",
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
        color: "#4a9080",
    },
};

Object.assign(styles, {
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
    typingBubble: {
        display: "flex",
        alignItems: "center",
        gap: "5px",
        padding: "10px 0",
    },
    dot: {
        width: "7px",
        height: "7px",
        borderRadius: "50%",
        background: "#b66a4e",
        display: "inline-block",
        animation: "typingPulse 1.2s ease-in-out infinite",
    },
});
