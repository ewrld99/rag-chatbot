import { useCallback, useEffect, useRef, useState } from "react";
import MessageBubble, { TypingBubble } from "./MessageBubble";
import ChatInput from "./ChatInput";
import { useSSEChat } from "../../hooks/useSSEChat";
import {
    createChatSession, deleteChatSession, getChatSession, getChatSessions, changePassword,
    getGenerationModels, submitMessageFeedback
} from "../../api/chatApi";
import udomLogo from "../../assets/udom-logo.svg";

// ─── Icons ────────────────────────────────────────────────────────────────────
const PlusIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
    </svg>
);
const MenuIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="18" x2="21" y2="18" />
    </svg>
);
const SidebarIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <line x1="9" y1="3" x2="9" y2="21"></line>
    </svg>
);
const XIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
);
const TrashIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="3 6 5 6 21 6" />
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
);
const UserIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
    </svg>
);
const UserPlusIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <line x1="19" y1="8" x2="19" y2="14" />
        <line x1="16" y1="11" x2="22" y2="11" />
    </svg>
);
const LoginIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
        <polyline points="10 17 15 12 10 7" />
        <line x1="15" y1="12" x2="3" y2="12" />
    </svg>
);
const LogOutIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
        <polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" />
    </svg>
);

const KeyIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"></path>
    </svg>
);
const InfoIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
    </svg>
);
const ChatLogo = () => (
    <img src={udomLogo} alt="The University of Dodoma" style={styles.logoImage} />
);

function navigate(to) {
    window.history.pushState({}, "", to);
    window.dispatchEvent(new PopStateEvent("popstate"));
}

const morningGreetings = [
    "Good morning",
    "Rise and shine",
    "Top of the morning",
    "A beautiful morning"
];

const afternoonGreetings = [
    "Good afternoon",
    "Hope your day is going well",
    "Good day"
];

const eveningGreetings = [
    "Good evening",
    "Welcome back",
    "Hope you had a great day"
];

const FALLBACK_MODEL_OPTIONS = [
    {
        id: "llama-3.3-70b-versatile",
        label: "Llama 3.3 70B Versatile",
        is_default: true,
    },
    { id: "openai/gpt-oss-120b", label: "GPT-OSS 120B" },
    { id: "qwen/qwen3.6-27b", label: "Qwen 3.6 27B" },
    { id: "openai/gpt-oss-20b", label: "GPT-OSS 20B" },
    { id: "llama-3.1-8b-instant", label: "Llama 3.1 8B Instant" },
];

function getTimeGreeting(date = new Date()) {
    const hour = date.getHours();
    const pickRandom = (arr) => arr[Math.floor(Math.random() * arr.length)];

    if (hour >= 5 && hour < 12) return pickRandom(morningGreetings);
    if (hour >= 12 && hour < 17) return pickRandom(afternoonGreetings);
    return pickRandom(eveningGreetings);
}

function capitalizeFirst(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
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

export default function ChatWindow() {
    const isNarrow = useMediaQuery("(max-width: 760px)");
    const [greeting] = useState(() => getTimeGreeting());
    const [user, setUser] = useState(() => {
        try {
            const saved = window.localStorage.getItem("ragUser");
            return saved ? JSON.parse(saved) : null;
        } catch {
            window.localStorage.removeItem("ragUser");
            return null;
        }
    });
    const [sessions, setSessions] = useState([]);
    const [activeSessionId, setActiveSessionId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isWaiting, setIsWaiting] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(() => !isNarrow);
    const [isAuthMenuOpen, setIsAuthMenuOpen] = useState(false);
    const [historyError, setHistoryError] = useState("");
    const [modelOptions, setModelOptions] = useState(FALLBACK_MODEL_OPTIONS);
    const [modelSelectionEnabled, setModelSelectionEnabled] = useState(true);
    const [modelPreference, setModelPreference] = useState(() => (
        window.localStorage.getItem("ragModelPreference") || "auto"
    ));
    const messageListRef = useRef(null);

    // Password Change State
    const [isPasswordModalOpen, setIsPasswordModalOpen] = useState(false);
    const [pwdOld, setPwdOld] = useState("");
    const [pwdNew, setPwdNew] = useState("");
    const [pwdError, setPwdError] = useState("");
    const [pwdSuccess, setPwdSuccess] = useState("");
    const [isChangingPwd, setIsChangingPwd] = useState(false);

    // Profile Modal State
    const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
    const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);

    const loadSessions = useCallback(async () => {
        if (!user) return;
        try {
            const data = await getChatSessions(user.id);
            setSessions(data);
            setHistoryError("");
        } catch (error) {
            setHistoryError(error.message);
        }
    }, [user]);

    useEffect(() => {
        let active = true;

        getGenerationModels()
            .then((policy) => {
                if (!active) return;
                const options = Array.isArray(policy?.models) ? policy.models : [];
                if (options.length > 0) setModelOptions(options);

                const selectionEnabled = policy?.selection_enabled !== false;
                setModelSelectionEnabled(selectionEnabled);
                setModelPreference((current) => {
                    const allowed = new Set(options.map((option) => option.id));
                    const next = (
                        selectionEnabled
                        && (current === "auto" || allowed.has(current))
                    ) ? current : "auto";
                    window.localStorage.setItem("ragModelPreference", next);
                    return next;
                });
            })
            .catch(() => {
                // Static allowlisted options remain available if policy loading fails.
            });

        return () => {
            active = false;
        };
    }, []);

    const handleModelChange = (event) => {
        const preference = event.target.value;
        setModelPreference(preference);
        window.localStorage.setItem("ragModelPreference", preference);
    };

    const handleSSEEvent = useCallback((event) => {
        if (event.type === "stream") {
            setIsWaiting(false);
            setMessages((prev) => {
                const last = prev[prev.length - 1];
                if (last?.role === "assistant") {
                    return [...prev.slice(0, -1), { ...last, content: last.content + event.token }];
                }
                return [...prev, { role: "assistant", content: event.token, sources: [] }];
            });
        }
        if (event.type === "done") {
            setIsWaiting(false);
            if (
                event.message_id
                || Array.isArray(event.sources)
                || event.selected_model
            ) {
                setMessages((prev) => {
                    const next = [...prev];
                    const lastMsg = next[next.length - 1];
                    if (lastMsg && lastMsg.role === "assistant") {
                        if (event.message_id) lastMsg.id = event.message_id;
                        if (Array.isArray(event.sources)) lastMsg.sources = event.sources;
                        if (event.selected_model) lastMsg.selectedModel = event.selected_model;
                        lastMsg.fallbackUsed = Boolean(event.fallback_used);
                    }
                    return next;
                });
            }
            loadSessions();
        }
        if (event.type === "error") {
            setIsWaiting(false);
            const safeMessage = event.message || "Unable to complete the answer right now. Please try again.";

            if (event.code === "GENERATION_TEMPORARILY_UNAVAILABLE") {
                setHistoryError("");
                setMessages((prev) => {
                    const sources = Array.isArray(event.sources) ? event.sources : [];
                    const last = prev[prev.length - 1];
                    if (last?.role === "assistant") {
                        return [
                            ...prev.slice(0, -1),
                            { ...last, content: safeMessage, sources },
                        ];
                    }
                    return [...prev, { role: "assistant", content: safeMessage, sources }];
                });
                return;
            }

            setHistoryError(safeMessage);
        }
    }, [loadSessions]);

    const { sendMessage } = useSSEChat(handleSSEEvent);

    useEffect(() => {
        const timer = window.setTimeout(() => loadSessions(), 0);
        return () => window.clearTimeout(timer);
    }, [loadSessions]);

    const handleLogout = () => {
        window.localStorage.removeItem("ragUser");
        setUser(null);
        setSessions([]);
        setActiveSessionId(null);
        setMessages([]);
    };

    const handleChangePasswordSubmit = async (e) => {
        e.preventDefault();
        setPwdError("");
        setPwdSuccess("");
        if (!pwdOld || !pwdNew) {
            setPwdError("Please fill in both fields.");
            return;
        }
        if (pwdNew.length < 6) {
            setPwdError("New password must be at least 6 characters.");
            return;
        }
        try {
            setIsChangingPwd(true);
            await changePassword(user.id, pwdOld, pwdNew);
            setPwdSuccess("Password updated successfully.");
            setPwdOld("");
            setPwdNew("");
            setTimeout(() => {
                setIsPasswordModalOpen(false);
                setPwdSuccess("");
            }, 2000);
        } catch (error) {
            setPwdError(error.message || "Failed to update password.");
        } finally {
            setIsChangingPwd(false);
        }
    };

    const handleNewChat = async () => {
        if (!user) { setActiveSessionId(null); setMessages([]); setHistoryError(""); return; }
        try {
            const session = await createChatSession(user.id);
            setActiveSessionId(session.id);
            setMessages([]);
            await loadSessions();
        } catch (error) {
            setHistoryError(error.message);
        }
    };

    const handleSelectSession = async (sessionId) => {
        if (!user) return;
        try {
            const session = await getChatSession(user.id, sessionId);
            setActiveSessionId(session.id);
            setMessages(session.messages.map((m) => ({
                id: m.id,
                role: m.role,
                content: m.content,
                sources: Array.isArray(m.sources) ? m.sources : [],
                feedback: m.feedback,
            })));
            setHistoryError("");
        } catch (error) {
            setHistoryError(error.message);
        }
    };

    const handleMessageFeedback = async (messageId, feedbackValue) => {
        if (!user) return; // Only logged-in users can give feedback (since DB logic expects valid messages which are tied to sessions, which are tied to users)
        
        // Optimistic update
        setMessages((prev) =>
            prev.map((m) => (m.id === messageId ? { ...m, feedback: feedbackValue } : m))
        );

        try {
            await submitMessageFeedback(messageId, feedbackValue);
        } catch (error) {
            console.error("Failed to submit feedback:", error);
            // Revert on failure could be implemented here
        }
    };

    const handleDeleteSession = async (e, sessionId) => {
        e.stopPropagation();
        if (!user) return;
        try {
            await deleteChatSession(user.id, sessionId);
            if (activeSessionId === sessionId) { setActiveSessionId(null); setMessages([]); }
            await loadSessions();
        } catch (error) {
            setHistoryError(error.message);
        }
    };

    const handleSend = async (text) => {
        let sessionId = activeSessionId;
        const recentHistory = messages
            .filter((message) => ["user", "assistant"].includes(message.role) && message.content?.trim())
            .slice(-12);

        setMessages((prev) => [...prev, { role: "user", content: text }]);
        setIsWaiting(true);
        if (user && !sessionId) {
            try {
                const session = await createChatSession(user.id, text.slice(0, 80));
                sessionId = session.id;
                setActiveSessionId(sessionId);
            } catch (error) {
                setHistoryError(error.message);
                setIsWaiting(false);
                return;
            }
        }
        // Pass userId + sessionId so the SSE endpoint can persist the exchange
        sendMessage(
            text,
            sessionId,
            user?.id ?? null,
            recentHistory,
            modelPreference,
        );
    };

    useEffect(() => {
        messageListRef.current?.scrollTo({ top: messageListRef.current.scrollHeight, behavior: "smooth" });
    }, [messages, isWaiting]);

    return (
        <div className="chat-window-theme" style={{ ...styles.chatWindow, ...(isNarrow ? styles.chatWindowNarrow : {}) }}>
            {/* Sidebar */}
            <aside className="chat-sidebar-theme" style={{
                ...styles.sidebar,
                ...(isSidebarOpen ? styles.sidebarOpen : styles.sidebarClosed),
                ...(isNarrow ? (isSidebarOpen ? styles.sidebarOpenNarrow : styles.sidebarClosedNarrow) : {}),
            }}>
                <div style={{ ...styles.sidebarInner, ...(isNarrow ? styles.sidebarInnerNarrow : {}) }}>
                    {/* Toggle button */}
                    <button
                        type="button"
                        style={styles.toggleBtn}
                        onClick={() => setIsSidebarOpen((v) => !v)}
                        aria-label={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
                    >
                        <SidebarIcon />
                    </button>

                    {isSidebarOpen && (
                        <div style={styles.sidebarContent}>
                            {/* Sidebar Header */}
                            <div style={styles.sidebarHeader}>
                                <div style={styles.brandMark}>
                                    <ChatLogo />
                                </div>
                                <div>
                                    <p style={styles.brandEyebrow}>UDOM Chatbot</p>
                                    <h2 style={styles.brandTitle}>Chats</h2>
                                </div>
                            </div>

                            {/* New Chat Button */}
                            <button type="button" className="new-chat-theme" style={styles.newChatBtn} onClick={handleNewChat}>
                                <PlusIcon />
                                New chat
                            </button>

                            {/* Auth / Sessions */}
                            <div style={styles.sessionSection}>
                                {user && (
                                    <>
                                        {historyError && <p style={styles.historyError}>{historyError}</p>}

                                        <div style={styles.historyList}>
                                            {sessions.length === 0 ? (
                                                <p style={styles.emptyHistory}>No saved chats yet.</p>
                                            ) : (
                                                sessions.map((session) => (
                                                    <div
                                                        key={session.id}
                                                        style={{
                                                            ...styles.historyItem,
                                                            ...(activeSessionId === session.id ? styles.historyItemActive : {}),
                                                        }}
                                                    >
                                                        <button
                                                            type="button"
                                                            style={styles.historyItemMain}
                                                            onClick={() => handleSelectSession(session.id)}
                                                        >
                                                            <span style={styles.historyItemTitle}>{session.title}</span>
                                                            <small style={styles.historyItemDate}>
                                                                {new Date(session.updated_at).toLocaleDateString()}
                                                            </small>
                                                        </button>
                                                        <button
                                                            type="button"
                                                            style={styles.historyItemDelete}
                                                            onClick={(e) => handleDeleteSession(e, session.id)}
                                                            aria-label="Delete chat"
                                                        >
                                                            <TrashIcon />
                                                        </button>
                                                    </div>
                                                ))
                                            )}
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* Bottom Card */}
                            {user ? (
                                <div style={styles.userCard} onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}>
                                    <div style={styles.userInfo}>
                                        <div style={styles.userAvatar}><UserIcon /></div>
                                        <span style={styles.userName}>{user.username}</span>
                                    </div>
                                    {isUserMenuOpen && (
                                        <div style={styles.userMenu}>
                                            <button type="button" style={styles.userMenuItem} onClick={(e) => { e.stopPropagation(); setIsUserMenuOpen(false); setIsProfileModalOpen(true); }}>
                                                <InfoIcon /> View Profile
                                            </button>
                                            <button type="button" style={styles.userMenuItem} onClick={(e) => { e.stopPropagation(); setIsUserMenuOpen(false); setIsPasswordModalOpen(true); }}>
                                                <KeyIcon /> Change Password
                                            </button>
                                            <button type="button" style={{ ...styles.userMenuItem, color: "var(--app-danger)" }} onClick={(e) => { e.stopPropagation(); setIsUserMenuOpen(false); handleLogout(); }}>
                                                <LogOutIcon /> Log Out
                                            </button>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div style={styles.guestAuthCard}>
                                    <div style={styles.guestAuthText}>
                                        <h3 style={styles.guestAuthTitle}>Sign in for more</h3>
                                        <p style={styles.guestAuthSubtitle}>Save your chat history, personalize your assistant, and more.</p>
                                    </div>
                                    <div style={styles.guestAuthButtons}>
                                        <button type="button" style={styles.guestLoginBtn} onClick={() => navigate("/login")}>Log in</button>
                                        <button type="button" style={styles.guestSignupBtn} onClick={() => navigate("/register")}>Create account</button>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </aside>

            {/* Main Chat Panel */}
            <div className="chat-panel-theme" style={{ ...styles.panel, ...(isNarrow ? styles.panelNarrow : {}) }}>
                {/* Panel Header */}
                <div className="chat-panel-header-theme" style={{ ...styles.panelHeader, ...(isNarrow ? styles.panelHeaderNarrow : {}) }}>
                    <div style={styles.panelHeaderLeft}>
                        {!isNarrow && <div style={styles.panelHeaderIcon}><ChatLogo /></div>}
                        <div>
                            <h2 style={styles.panelTitle}>University Assistant</h2>
                            <p style={styles.panelSubtitle}>
                                {user ? capitalizeFirst(user.username) : "Ready to help"}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Messages */}
                <div className="chat-message-list-theme" style={{ ...styles.messageList, ...(isNarrow ? styles.messageListNarrow : {}) }} ref={messageListRef}>
                    {messages.length === 0 ? (
                        <section style={{ ...styles.emptyChat, ...(isNarrow ? styles.emptyChatNarrow : {}) }}>
                            <div style={styles.emptyChatIcon}><ChatLogo /></div>
                            <h3 style={styles.emptyChatTitle}>
                                {user ? `${greeting}, ${capitalizeFirst(user.username)}` : "How can I assist you?"}
                            </h3>
                            <p style={styles.emptyChatSubtitle}>
                                {user
                                    ? "Ask questions and get clear, helpful answers. Your chat is saved."
                                    : "Ask questions and get clear, helpful answers. Log in only if you want saved history."}
                            </p>
                            <ChatInput
                                onSend={handleSend}
                                disabled={isWaiting}
                                placement="center"
                                modelOptions={modelOptions}
                                modelPreference={modelPreference}
                                modelSelectionEnabled={modelSelectionEnabled}
                                onModelChange={handleModelChange}
                            />
                        </section>
                    ) : (
                        messages.map((m, i) => (
                            <MessageBubble 
                                key={`${m.role}-${i}`} 
                                {...m} 
                                onFeedback={handleMessageFeedback}
                            />
                        ))
                    )}

                    {isWaiting && <TypingBubble />}
                </div>

                {messages.length > 0 && (
                    <ChatInput
                        onSend={handleSend}
                        disabled={isWaiting}
                        modelOptions={modelOptions}
                        modelPreference={modelPreference}
                        modelSelectionEnabled={modelSelectionEnabled}
                        onModelChange={handleModelChange}
                    />
                )}
            </div>

            {/* Profile Modal */}
            {isProfileModalOpen && user && (
                <div style={styles.modalOverlay}>
                    <div style={styles.modalContent}>
                        <div style={styles.modalHeader}>
                            <h3 style={styles.modalTitle}>Your Profile</h3>
                            <button type="button" style={styles.modalCloseBtn} onClick={() => setIsProfileModalOpen(false)}>
                                <XIcon />
                            </button>
                        </div>
                        <div style={styles.modalBody}>
                            <div style={{ marginBottom: "15px" }}>
                                <label style={{ display: "block", fontSize: "12px", color: "var(--app-text-muted)", marginBottom: "4px" }}>Username</label>
                                <div style={{ fontSize: "14px", fontWeight: "500", color: "var(--app-text)" }}>{user.username}</div>
                            </div>
                            <div style={{ marginBottom: "15px" }}>
                                <label style={{ display: "block", fontSize: "12px", color: "var(--app-text-muted)", marginBottom: "4px" }}>Registration Number</label>
                                <div style={{ fontSize: "14px", fontWeight: "500", color: "var(--app-text)" }}>{user.registration_number || "Not provided"}</div>
                            </div>
                            <div style={{ marginBottom: "15px" }}>
                                <label style={{ display: "block", fontSize: "12px", color: "var(--app-text-muted)", marginBottom: "4px" }}>Programme</label>
                                <div style={{ fontSize: "14px", fontWeight: "500", color: "var(--app-text)" }}>{user.programme || "Not provided"}</div>
                            </div>
                            <div style={{ marginBottom: "15px" }}>
                                <label style={{ display: "block", fontSize: "12px", color: "var(--app-text-muted)", marginBottom: "4px" }}>Campus</label>
                                <div style={{ fontSize: "14px", fontWeight: "500", color: "var(--app-text)" }}>{user.campus || "Not provided"}</div>
                            </div>
                            <div style={{ marginBottom: "15px" }}>
                                <label style={{ display: "block", fontSize: "12px", color: "var(--app-text-muted)", marginBottom: "4px" }}>Admission Year</label>
                                <div style={{ fontSize: "14px", fontWeight: "500", color: "var(--app-text)" }}>{user.admission_year || "Not provided"}</div>
                            </div>
                        </div>
                        <div style={styles.modalFooter}>
                            <button type="button" style={styles.modalCancelBtn} onClick={() => setIsProfileModalOpen(false)}>
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Change Password Modal */}
            {isPasswordModalOpen && (
                <div style={styles.modalOverlay}>
                    <div style={styles.modalContent}>
                        <div style={styles.modalHeader}>
                            <h3 style={styles.modalTitle}>Change Password</h3>
                            <button type="button" style={styles.modalCloseBtn} onClick={() => setIsPasswordModalOpen(false)}>
                                <XIcon />
                            </button>
                        </div>
                        <form onSubmit={handleChangePasswordSubmit} style={styles.modalForm}>
                            <input
                                type="password"
                                placeholder="Current Password"
                                value={pwdOld}
                                onChange={(e) => setPwdOld(e.target.value)}
                                style={styles.modalInput}
                                required
                            />
                            <input
                                type="password"
                                placeholder="New Password (min 6 chars)"
                                value={pwdNew}
                                onChange={(e) => setPwdNew(e.target.value)}
                                style={styles.modalInput}
                                required
                            />
                            {pwdError && <p style={styles.modalError}>{pwdError}</p>}
                            {pwdSuccess && <p style={styles.modalSuccess}>{pwdSuccess}</p>}
                            <div style={styles.modalActions}>
                                <button type="button" onClick={() => setIsPasswordModalOpen(false)} style={styles.modalCancelBtn}>
                                    Cancel
                                </button>
                                <button type="submit" disabled={isChangingPwd} style={styles.modalSubmitBtn}>
                                    {isChangingPwd ? "Updating..." : "Update Password"}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}

const styles = {
    chatWindow: {
        display: "flex",
        height: "100vh",
        background: "#0f1923",
        overflow: "hidden",
        fontFamily: "inherit",
    },
    chatWindowNarrow: {
        flexDirection: "column",
        height: "100dvh",
        minHeight: 0,
    },
    sidebar: {
        flexShrink: 0,
        transition: "width 0.25s ease",
        overflow: "hidden",
        background: "#131e2d",
        borderRight: "1px solid rgba(255,255,255,0.06)",
    },
    sidebarOpen: { width: "280px" },
    sidebarClosed: { width: "56px" },
    sidebarOpenNarrow: {
        width: "100%",
        maxHeight: "42dvh",
    },
    sidebarClosedNarrow: {
        width: "100%",
        height: "60px",
    },
    sidebarInner: {
        width: "100%",
        height: "100%",
        flex: 1,
        minHeight: 0,
        display: "flex",
        flexDirection: "column",
        position: "relative",
    },
    sidebarInnerNarrow: {
        minWidth: 0,
    },
    toggleBtn: {
        width: "36px",
        height: "36px",
        margin: "12px 10px",
        borderRadius: "10px",
        border: "1px solid rgba(255,255,255,0.08)",
        background: "transparent",
        color: "#718096",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        flexShrink: 0,
        transition: "all 0.15s",
    },
    sidebarContent: {
        flex: 1,
        minHeight: 0,
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        padding: "0 12px 16px",
        overflow: "hidden",
    },
    sidebarHeader: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "4px 0 8px",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        marginBottom: "4px",
    },
    brandMark: {
        width: "36px",
        height: "36px",
        borderRadius: "50%",
        background: "#050505",
        border: "1px solid rgba(255,255,255,0.16)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        overflow: "hidden",
    },
    logoImage: {
        width: "100%",
        height: "100%",
        objectFit: "cover",
        borderRadius: "50%",
        display: "block",
    },
    brandEyebrow: {
        fontSize: "10px",
        fontWeight: "700",
        letterSpacing: "1.2px",
        textTransform: "uppercase",
        color: "#63b3a4",
        margin: 0,
    },
    brandTitle: {
        fontSize: "16px",
        fontWeight: "700",
        color: "#e2e8f0",
        margin: 0,
    },
    newChatBtn: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "9px 12px",
        borderRadius: "10px",
        border: "1px solid rgba(99,179,164,0.25)",
        background: "rgba(99,179,164,0.07)",
        color: "#63b3a4",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        width: "100%",
        transition: "all 0.15s",
    },
    guestSection: {
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        flex: 1,
    },
    openAuthBtn: {
        padding: "9px 12px",
        borderRadius: "10px",
        border: "1px solid rgba(255,255,255,0.1)",
        background: "transparent",
        color: "#a0aec0",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        width: "100%",
        textAlign: "left",
    },
    authCard: {
        background: "#1a2a3d",
        borderRadius: "14px",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        border: "1px solid rgba(255,255,255,0.07)",
    },
    authCardHeader: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
    },
    authCardTitle: {
        fontSize: "14px",
        fontWeight: "700",
        color: "#e2e8f0",
        margin: 0,
    },
    closeAuthBtn: {
        width: "26px",
        height: "26px",
        borderRadius: "6px",
        border: "1px solid rgba(255,255,255,0.07)",
        background: "transparent",
        color: "#718096",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
    },
    authField: { display: "flex", flexDirection: "column", gap: "4px" },
    authLabel: { fontSize: "11px", fontWeight: "600", color: "#718096" },
    authInput: {
        padding: "8px 10px",
        borderRadius: "8px",
        border: "1px solid rgba(255,255,255,0.08)",
        background: "rgba(255,255,255,0.04)",
        color: "#e2e8f0",
        fontSize: "13px",
        outline: "none",
        transition: "border-color 0.2s",
        width: "100%",
        boxSizing: "border-box",
    },
    authInputFocused: {
        borderColor: "rgba(99,179,164,0.4)",
        boxShadow: "0 0 0 2px rgba(99,179,164,0.08)",
    },
    authError: {
        fontSize: "12px",
        color: "#fc814a",
        background: "rgba(252,129,74,0.08)",
        border: "1px solid rgba(252,129,74,0.2)",
        borderRadius: "8px",
        padding: "8px 10px",
        margin: 0,
    },
    authSubmitBtn: {
        padding: "9px",
        borderRadius: "8px",
        border: "none",
        background: "linear-gradient(135deg, #63b3a4, #4a9080)",
        color: "#0d1520",
        fontSize: "13px",
        fontWeight: "700",
        cursor: "pointer",
        width: "100%",
    },
    authToggleBtn: {
        background: "transparent",
        border: "none",
        color: "#63b3a4",
        fontSize: "12px",
        cursor: "pointer",
        padding: "0",
        textAlign: "center",
        width: "100%",
    },
    sessionSection: {
        display: "flex",
        flexDirection: "column",
        flex: 1,
        minHeight: 0,
        gap: "8px",
        overflow: "hidden",
    },
    historyError: {
        fontSize: "12px",
        color: "#fc814a",
        background: "rgba(252,129,74,0.08)",
        borderRadius: "8px",
        padding: "8px 10px",
        margin: 0,
    },
    historyList: {
        flex: 1,
        minHeight: 0,
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: "4px",
    },
    emptyHistory: {
        fontSize: "12px",
        color: "#4a5568",
        textAlign: "center",
        padding: "24px 0",
        margin: 0,
    },
    historyItem: {
        display: "flex",
        alignItems: "center",
        borderRadius: "10px",
        overflow: "hidden",
        border: "1px solid transparent",
        transition: "all 0.15s",
        flexShrink: 0,
        minHeight: "48px",
    },
    historyItemActive: {
        background: "rgba(99,179,164,0.08)",
        border: "1px solid rgba(99,179,164,0.2)",
    },
    historyItemMain: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        gap: "2px",
        padding: "9px 10px",
        background: "transparent",
        border: "none",
        cursor: "pointer",
        textAlign: "left",
        overflow: "hidden",
    },
    historyItemTitle: {
        fontSize: "12px",
        fontWeight: "600",
        color: "#a0aec0",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        display: "block",
    },
    historyItemDate: {
        fontSize: "10px",
        color: "#4a5568",
    },
    historyItemDelete: {
        width: "32px",
        height: "32px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "transparent",
        border: "none",
        color: "#4a5568",
        cursor: "pointer",
        flexShrink: 0,
        borderRadius: "8px",
        margin: "2px",
        transition: "all 0.15s",
    },
    userCard: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "10px 12px",
        borderRadius: "10px",
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.06)",
        marginTop: "auto",
    },
    userInfo: { display: "flex", alignItems: "center", gap: "8px" },
    userAvatar: {
        width: "28px",
        height: "28px",
        borderRadius: "8px",
        background: "rgba(99,179,164,0.1)",
        border: "1px solid rgba(99,179,164,0.2)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#63b3a4",
    },
    userName: { fontSize: "12px", fontWeight: "600", color: "#a0aec0" },
    logoutBtn: {
        width: "28px",
        height: "28px",
        borderRadius: "8px",
        border: "1px solid rgba(255,255,255,0.07)",
        background: "transparent",
        color: "#4a5568",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
    },
    panel: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        minWidth: 0,
        overflow: "hidden",
    },
    panelNarrow: {
        minHeight: 0,
    },
    panelHeader: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "14px 20px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "#131e2d",
        gap: "12px",
    },
    panelHeaderNarrow: {
        alignItems: "center",
        padding: "12px 14px",
    },
    panelHeaderLeft: { display: "flex", alignItems: "center", gap: "12px" },
    panelHeaderIcon: {
        width: "42px",
        height: "42px",
        borderRadius: "50%",
        background: "#050505",
        border: "1px solid rgba(255,255,255,0.16)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
    },
    panelTitle: { fontSize: "16px", fontWeight: "700", color: "#e2e8f0", margin: 0 },
    panelSubtitle: { fontSize: "12px", color: "#4a5568", margin: 0 },
    connectionPill: {
        padding: "4px 12px",
        borderRadius: "20px",
        fontSize: "11px",
        fontWeight: "700",
        letterSpacing: "0.3px",
        whiteSpace: "nowrap",
    },
    connectionPillNarrow: {
        padding: "4px 8px",
        fontSize: "10px",
    },
    pillActive: {
        background: "rgba(99,179,164,0.1)",
        border: "1px solid rgba(99,179,164,0.25)",
        color: "#63b3a4",
    },
    pillGuest: {
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.08)",
        color: "#4a5568",
    },
    messageList: {
        flex: 1,
        overflowY: "auto",
        padding: "20px 20px 8px",
        display: "flex",
        flexDirection: "column",
        gap: "4px",
        minHeight: 0,
    },
    messageListNarrow: {
        padding: "14px 12px 8px",
    },
    emptyChat: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        flex: 1,
        textAlign: "center",
        gap: "12px",
        padding: "40px 24px",
        margin: "auto 0",
    },
    emptyChatNarrow: {
        justifyContent: "flex-start",
        padding: "20px 4px",
        margin: 0,
    },
    emptyChatIcon: {
        width: "92px",
        height: "92px",
        borderRadius: "50%",
        background: "#050505",
        border: "1px solid rgba(255,255,255,0.16)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: "4px",
        overflow: "hidden",
    },
    emptyChatTitle: { fontSize: "20px", fontWeight: "700", color: "#e2e8f0", margin: 0 },
    emptyChatSubtitle: { fontSize: "14px", color: "#718096", margin: 0, maxWidth: "400px", lineHeight: 1.5 },
    promptGrid: {
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        width: "100%",
        maxWidth: "420px",
        marginTop: "8px",
    },
    promptGridNarrow: {
        maxWidth: "100%",
    },
    promptBtn: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "12px 16px",
        borderRadius: "12px",
        border: "1px solid rgba(255,255,255,0.07)",
        background: "rgba(255,255,255,0.025)",
        color: "#a0aec0",
        fontSize: "13px",
        cursor: "pointer",
        textAlign: "left",
        transition: "all 0.15s",
    },
    promptIcon: {
        color: "#63b3a4",
        flexShrink: 0,
        display: "flex",
    },
};

Object.assign(styles, {
    chatWindow: {
        display: "flex",
        height: "100vh",
        background: "var(--app-bg)",
        overflow: "hidden",
        fontFamily: "inherit",
        color: "var(--app-text)",
    },
    chatWindowNarrow: {
        flexDirection: "column",
        height: "100dvh",
        minHeight: 0,
    },
    sidebar: {
        flexShrink: 0,
        alignSelf: "stretch",
        height: "100%",
        transition: "width 0.25s ease",
        overflow: "hidden",
        background: "var(--app-bg)",
        borderRight: "1px solid var(--app-border)",
    },
    sidebarOpen: { width: "280px" },
    sidebarClosed: { width: "64px" },
    sidebarOpenNarrow: {
        width: "100%",
        height: "100%",
        maxHeight: "none",
    },
    sidebarClosedNarrow: {
        width: "100%",
        height: "60px",
    },
    toggleBtn: {
        width: "36px",
        height: "36px",
        margin: "12px 10px",
        borderRadius: "10px",
        border: "1px solid var(--app-border)",
        background: "transparent",
        color: "var(--app-muted)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        flexShrink: 0,
        transition: "all 0.15s",
    },
    sidebarHeader: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "4px 0 8px",
        borderBottom: "1px solid var(--app-border)",
        marginBottom: "4px",
    },
    brandMark: {
        width: "36px",
        height: "36px",
        borderRadius: "50%",
        background: "#050505",
        border: "1px solid var(--app-border-strong)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        overflow: "hidden",
    },
    brandEyebrow: {
        fontSize: "10px",
        fontWeight: "700",
        letterSpacing: "1.2px",
        textTransform: "uppercase",
        color: "var(--app-accent)",
        margin: 0,
    },
    brandTitle: {
        fontSize: "16px",
        fontWeight: "700",
        color: "var(--app-text)",
        margin: 0,
    },
    newChatBtn: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "9px 12px",
        borderRadius: "8px",
        border: "1px solid var(--app-border)",
        background: "var(--app-surface-muted)",
        color: "var(--app-text)",
        fontSize: "13px",
        fontWeight: "700",
        cursor: "pointer",
        width: "100%",
        transition: "all 0.15s",
    },
    openAuthBtn: {
        padding: "9px 12px",
        borderRadius: "8px",
        border: "1px solid var(--app-border-strong)",
        background: "transparent",
        color: "var(--app-text)",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        width: "100%",
        textAlign: "left",
    },
    authCard: {
        background: "var(--app-surface-muted)",
        borderRadius: "8px",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        border: "1px solid var(--app-border)",
    },
    authCardTitle: {
        fontSize: "14px",
        fontWeight: "700",
        color: "var(--app-text)",
        margin: 0,
    },
    guestAuthCard: {
        marginTop: "auto",
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        padding: "16px",
        background: "var(--app-surface-muted)",
        borderRadius: "12px",
        border: "1px solid var(--app-border)",
    },
    guestAuthText: {
        display: "flex",
        flexDirection: "column",
        gap: "4px",
    },
    guestAuthTitle: {
        fontSize: "14px",
        fontWeight: "700",
        color: "var(--app-text)",
        margin: 0,
    },
    guestAuthSubtitle: {
        fontSize: "12px",
        color: "var(--app-muted)",
        margin: 0,
        lineHeight: 1.4,
    },
    guestAuthButtons: {
        display: "flex",
        flexDirection: "column",
        gap: "8px",
    },
    guestLoginBtn: {
        padding: "10px",
        borderRadius: "8px",
        border: "1px solid var(--app-border-strong)",
        background: "transparent",
        color: "var(--app-text)",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        textAlign: "center",
        transition: "all 0.15s",
    },
    guestSignupBtn: {
        padding: "10px",
        borderRadius: "8px",
        border: "1px solid var(--app-text)",
        background: "var(--app-text)",
        color: "var(--app-surface)",
        fontSize: "13px",
        fontWeight: "600",
        cursor: "pointer",
        textAlign: "center",
        transition: "all 0.15s",
    },
    sessionSection: {
        display: "flex",
        flexDirection: "column",
        flex: 1,
        minHeight: 0,
        overflow: "hidden",
    },
    closeAuthBtn: {
        width: "26px",
        height: "26px",
        borderRadius: "8px",
        border: "1px solid var(--app-border)",
        background: "transparent",
        color: "var(--app-muted)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
    },
    authLabel: { fontSize: "11px", fontWeight: "600", color: "var(--app-muted)" },
    authInput: {
        padding: "8px 10px",
        borderRadius: "8px",
        border: "1px solid var(--app-border-strong)",
        background: "#ffffff",
        color: "var(--app-text)",
        fontSize: "13px",
        outline: "none",
        transition: "border-color 0.2s, box-shadow 0.2s",
        width: "100%",
        boxSizing: "border-box",
    },
    authInputFocused: {
        borderColor: "var(--app-accent-strong)",
        boxShadow: "0 0 0 3px rgba(217,108,71,0.12)",
    },
    authError: {
        fontSize: "12px",
        color: "var(--app-danger)",
        background: "var(--app-danger-soft)",
        border: "1px solid #f1c4b2",
        borderRadius: "8px",
        padding: "8px 10px",
        margin: 0,
    },
    authSubmitBtn: {
        padding: "9px",
        borderRadius: "8px",
        border: "none",
        background: "var(--app-text)",
        color: "var(--app-surface-muted)",
        fontSize: "13px",
        fontWeight: "700",
        cursor: "pointer",
        width: "100%",
    },
    authToggleBtn: {
        background: "transparent",
        border: "none",
        color: "var(--app-accent)",
        fontSize: "12px",
        cursor: "pointer",
        padding: "0",
        textAlign: "center",
        width: "100%",
    },
    historyError: {
        fontSize: "12px",
        color: "var(--app-danger)",
        background: "var(--app-danger-soft)",
        borderRadius: "8px",
        padding: "8px 10px",
        margin: 0,
    },
    emptyHistory: {
        fontSize: "12px",
        color: "var(--app-faint)",
        textAlign: "center",
        padding: "24px 0",
        margin: 0,
    },
    historyItem: {
        display: "flex",
        alignItems: "center",
        borderRadius: "10px",
        overflow: "hidden",
        border: "1px solid transparent",
        transition: "all 0.2s ease",
        marginBottom: "4px",
        cursor: "pointer",
        flexShrink: 0,
        minHeight: "48px",
    },
    historyItemActive: {
        background: "var(--app-panel)",
        border: "1px solid var(--app-border-strong)",
        boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
    },
    historyItemMain: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        padding: "10px 12px",
        background: "transparent",
        border: "none",
        cursor: "pointer",
        textAlign: "left",
        overflow: "hidden",
    },
    historyItemTitle: {
        fontSize: "13px",
        fontWeight: "600",
        color: "var(--app-text)",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        display: "block",
        width: "100%",
    },
    historyItemDate: {
        fontSize: "11px",
        color: "var(--app-muted)",
        marginTop: "3px",
    },
    historyItemDelete: {
        width: "28px",
        height: "28px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--app-danger-soft)",
        border: "none",
        color: "var(--app-danger)",
        cursor: "pointer",
        flexShrink: 0,
        borderRadius: "6px",
        marginRight: "8px",
        opacity: 0.8,
        transition: "all 0.2s",
    },
    userCard: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "10px 12px",
        borderRadius: "8px",
        background: "var(--app-surface-muted)",
        border: "1px solid var(--app-border)",
        marginTop: "auto",
        position: "relative",
        cursor: "pointer",
        transition: "background 0.15s",
    },
    userAvatar: {
        width: "28px",
        height: "28px",
        borderRadius: "8px",
        background: "var(--app-surface-muted)",
        border: "1px solid #d9c9b4",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--app-accent)",
    },
    userName: { fontSize: "12px", fontWeight: "600", color: "#4d4942" },
    userMenu: {
        position: "absolute",
        bottom: "calc(100% + 8px)",
        left: 0,
        right: 0,
        background: "var(--app-surface-muted)",
        borderRadius: "12px",
        padding: "6px",
        boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
        border: "1px solid var(--app-border)",
        zIndex: 100,
        display: "flex",
        flexDirection: "column",
        gap: "2px",
    },
    userMenuItem: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "10px",
        borderRadius: "8px",
        border: "none",
        background: "transparent",
        color: "var(--app-text)",
        fontSize: "13px",
        fontWeight: "500",
        cursor: "pointer",
        textAlign: "left",
        width: "100%",
    },
    panel: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        minWidth: 0,
        overflow: "hidden",
        background: "var(--app-bg)",
    },
    panelHeader: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "14px 24px",
        borderBottom: "1px solid var(--app-border)",
        background: "var(--app-surface)",
        backdropFilter: "blur(14px)",
        gap: "12px",
    },
    authHeaderActions: {
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-end",
        gap: "6px",
        marginLeft: "auto",
        marginRight: "58px",
        padding: "5px",
        border: "1px solid var(--app-border)",
        borderRadius: "16px",
        background: "var(--app-surface)",
        flexShrink: 0,
        position: "relative",
        boxShadow: "0 10px 26px rgba(72, 61, 47, 0.08)",
    },
    authHeaderActionsNarrow: {
        width: "auto",
        justifyContent: "flex-end",
        marginLeft: "auto",
        marginRight: "52px",
        padding: "6px",
    },
    accountMenuButton: {
        width: "38px",
        height: "38px",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: "12px",
        border: "1px solid var(--app-border-strong)",
        background: "var(--app-surface-muted)",
        color: "var(--app-text)",
        cursor: "pointer",
        transition: "border-color 0.15s, background 0.15s, transform 0.15s",
    },
    accountMenu: {
        position: "absolute",
        top: "calc(100% + 10px)",
        right: 0,
        zIndex: 20,
        width: "210px",
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        padding: "8px",
        borderRadius: "14px",
        border: "1px solid var(--app-border)",
        background: "var(--app-surface)",
        boxShadow: "0 18px 42px rgba(72, 61, 47, 0.16)",
    },
    accountMenuItem: {
        minHeight: "38px",
        display: "flex",
        alignItems: "center",
        gap: "9px",
        width: "100%",
        borderRadius: "10px",
        padding: "9px 11px",
        fontSize: "13px",
        fontWeight: "800",
        cursor: "pointer",
        textAlign: "left",
    },
    accountMenuSecondary: {
        border: "1px solid transparent",
        background: "transparent",
        color: "var(--app-text)",
    },
    accountMenuPrimary: {
        border: "1px solid var(--app-text)",
        background: "var(--app-text)",
        color: "var(--app-surface-muted)",
    },
    headerAuthButton: {
        minHeight: "36px",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "8px",
        padding: "9px 13px",
        borderRadius: "10px",
        fontSize: "13px",
        fontWeight: "700",
        cursor: "pointer",
        whiteSpace: "nowrap",
        transition: "border-color 0.15s, background 0.15s, color 0.15s, transform 0.15s",
    },
    headerLoginButton: {
        border: "1px solid transparent",
        background: "transparent",
        color: "var(--app-muted)",
    },
    headerCreateButton: {
        border: "1px solid var(--app-text)",
        background: "var(--app-text)",
        color: "var(--app-surface-muted)",
    },
    panelHeaderIcon: {
        width: "46px",
        height: "46px",
        borderRadius: "50%",
        background: "var(--app-surface-muted)",
        border: "1px solid var(--app-border-strong)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
    },
    panelTitle: { fontSize: "17px", fontWeight: "800", color: "var(--app-text)", margin: 0, lineHeight: 1.1 },
    panelSubtitle: { fontSize: "12px", color: "var(--app-faint)", margin: "3px 0 0" },
    pillActive: {
        background: "var(--app-accent-soft)",
        border: "1px solid var(--app-border-strong)",
        color: "var(--app-accent)",
    },
    pillGuest: {
        background: "var(--app-bg)",
        border: "1px solid var(--app-border)",
        color: "var(--app-muted)",
    },
    messageList: {
        flex: 1,
        overflowY: "auto",
        padding: "24px max(24px, calc((100vw - 940px) / 2)) 12px",
        display: "flex",
        flexDirection: "column",
        gap: "2px",
        minHeight: 0,
    },
    messageListNarrow: {
        padding: "16px 14px 8px",
    },
    emptyChat: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        flex: 1,
        textAlign: "center",
        gap: "14px",
        padding: "48px 24px",
        margin: "auto 0",
    },
    emptyChatIcon: {
        width: "92px",
        height: "92px",
        borderRadius: "50%",
        background: "var(--app-surface-muted)",
        border: "1px solid var(--app-border-strong)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: "4px",
        overflow: "hidden",
    },
    emptyChatTitle: {
        fontSize: "clamp(30px, 5vw, 44px)",
        fontWeight: "700",
        color: "var(--app-text)",
        margin: 0,
        letterSpacing: "-0.02em",
    },
    emptyChatSubtitle: {
        fontSize: "15px",
        color: "var(--app-muted)",
        margin: 0,
        maxWidth: "460px",
        lineHeight: 1.55,
    },
    promptGrid: {
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        width: "100%",
        maxWidth: "620px",
        marginTop: "8px",
    },
    promptBtn: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "12px 16px",
        borderRadius: "12px",
        border: "1px solid var(--app-border)",
        background: "var(--app-surface-muted)",
        color: "var(--app-text)",
        fontSize: "14px",
        cursor: "pointer",
        textAlign: "left",
        transition: "all 0.15s",
    },
    promptIcon: {
        color: "var(--app-accent)",
        flexShrink: 0,
        display: "flex",
    },
    modalOverlay: {
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0,0,0,0.5)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: "20px",
    },
    modalContent: {
        backgroundColor: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        borderRadius: "16px",
        padding: "24px",
        width: "100%",
        maxWidth: "400px",
        boxShadow: "0 24px 48px rgba(0,0,0,0.4)",
    },
    modalHeader: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: "20px",
    },
    modalTitle: {
        fontSize: "18px",
        fontWeight: "600",
        color: "var(--app-text)",
        margin: 0,
    },
    modalCloseBtn: {
        background: "none",
        border: "none",
        color: "var(--app-text-muted)",
        cursor: "pointer",
        display: "flex",
        padding: "4px",
        borderRadius: "6px",
        transition: "all 0.2s",
    },
    modalForm: {
        display: "flex",
        flexDirection: "column",
        gap: "16px",
    },
    modalInput: {
        width: "100%",
        backgroundColor: "var(--app-bg)",
        border: "1px solid var(--app-border)",
        color: "var(--app-text)",
        padding: "12px 14px",
        borderRadius: "8px",
        fontSize: "14px",
        outline: "none",
        transition: "border-color 0.2s",
    },
    modalError: {
        color: "var(--app-danger)",
        fontSize: "13px",
        margin: 0,
    },
    modalSuccess: {
        color: "var(--app-success)",
        fontSize: "13px",
        margin: 0,
    },
    modalActions: {
        display: "flex",
        justifyContent: "flex-end",
        gap: "12px",
        marginTop: "8px",
    },
    modalCancelBtn: {
        padding: "10px 16px",
        borderRadius: "8px",
        background: "transparent",
        color: "var(--app-text)",
        border: "1px solid var(--app-border)",
        cursor: "pointer",
        fontSize: "14px",
        fontWeight: "500",
    },
    modalSubmitBtn: {
        padding: "10px 16px",
        borderRadius: "8px",
        background: "var(--app-primary)",
        color: "#fff",
        border: "none",
        cursor: "pointer",
        fontSize: "14px",
        fontWeight: "500",
        transition: "background 0.2s",
    },
});
