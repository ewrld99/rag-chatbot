import { useEffect, useState } from "react";

const SendIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <line x1="22" y1="2" x2="11" y2="13" />
        <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
);

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

export default function ChatInput({ onSend, disabled = false, placement = "bottom" }) {
    const isNarrow = useMediaQuery("(max-width: 760px)");
    const [text, setText] = useState("");
    const [isFocused, setIsFocused] = useState(false);
    const isCentered = placement === "center";

    const handleSend = () => {
        const message = text.trim();
        if (!message || disabled) return;
        onSend(message);
        setText("");
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const canSend = text.trim().length > 0 && !disabled;

    return (
        <div
            style={{
                ...styles.wrapper,
                ...(isCentered ? styles.wrapperCentered : {}),
                ...(isNarrow ? styles.wrapperNarrow : {}),
            }}
        >
            <div style={{
                ...styles.inputRow,
                ...(isFocused ? styles.inputRowFocused : {}),
                ...(isNarrow ? styles.inputRowNarrow : {}),
            }}>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    placeholder="Ask anything..."
                    rows={1}
                    disabled={disabled}
                    style={{ ...styles.textarea, ...(isNarrow ? styles.textareaNarrow : {}) }}
                />
                <button
                    type="button"
                    onClick={handleSend}
                    disabled={!canSend}
                    aria-label="Send message"
                    style={{
                        ...styles.sendBtn,
                        ...(canSend ? styles.sendBtnActive : styles.sendBtnDisabled),
                    }}
                >
                    <SendIcon />
                </button>
            </div>
            <p
                style={{
                    ...styles.hint,
                    ...(isCentered ? styles.hintCentered : {}),
                    ...(isNarrow ? styles.hintNarrow : {}),
                }}
            >
                Press Enter to send / Shift + Enter for new line
            </p>
        </div>
    );
}

const styles = {
    wrapper: {
        padding: "12px 24px 22px",
        background: "transparent",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
    },
    wrapperNarrow: {
        padding: "10px 12px 14px",
    },
    wrapperCentered: {
        width: "min(820px, 100%)",
        padding: "6px 0 0",
        background: "transparent",
    },
    inputRow: {
        width: "min(780px, 100%)",
        margin: "0 auto",
        display: "flex",
        alignItems: "flex-end",
        gap: "12px",
        background: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        borderRadius: "24px",
        padding: "10px 12px 10px 20px",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.05)",
        transition: "border-color 0.2s, box-shadow 0.2s",
    },
    inputRowNarrow: {
        gap: "8px",
        padding: "8px 10px 8px 16px",
        borderRadius: "20px",
    },
    inputRowFocused: {
        borderColor: "var(--app-accent)",
        boxShadow: "0 0 0 4px color-mix(in srgb, var(--app-accent) 15%, transparent), 0 8px 32px rgba(0, 0, 0, 0.05)",
    },
    textarea: {
        flex: 1,
        background: "transparent",
        border: "none",
        outline: "none",
        resize: "none",
        fontSize: "15px",
        lineHeight: "1.55",
        color: "var(--app-text)",
        maxHeight: "160px",
        overflowY: "auto",
        padding: "6px 0",
        fontFamily: "inherit",
    },
    textareaNarrow: {
        fontSize: "14px",
        maxHeight: "112px",
    },
    sendBtn: {
        width: "38px",
        height: "38px",
        borderRadius: "12px",
        border: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        flexShrink: 0,
        transition: "all 0.15s ease",
    },
    sendBtnActive: {
        background: "var(--app-text)",
        color: "var(--app-surface)",
        boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
    },
    sendBtnDisabled: {
        background: "var(--app-surface-muted)",
        color: "var(--app-muted)",
        cursor: "not-allowed",
    },
    hint: {
        width: "min(780px, 100%)",
        margin: "0 auto",
        fontSize: "11px",
        color: "var(--app-faint)",
        textAlign: "center",
        paddingTop: "2px",
    },
    hintCentered: {
        color: "var(--app-faint)",
    },
    hintNarrow: {
        display: "none",
    },
};
