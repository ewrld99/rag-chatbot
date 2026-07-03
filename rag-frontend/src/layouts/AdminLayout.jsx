import { useEffect, useState } from "react";
import NavigationLink from "../components/NavigationLink.jsx";
import { clearAdminSession, getAdminSession } from "../utils/adminAuth.js";
import { changePassword } from "../api/chatApi.js";
import udomLogo from "../assets/udom-logo.svg";

const DashboardIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" />
    </svg>
);

const UploadIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
    </svg>
);

const DocumentIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
        <path d="M14 2v4a2 2 0 0 0 2 2h4" /><path d="M10 9H8" /><path d="M16 13H8" /><path d="M16 17H8" />
    </svg>
);

const VectorIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
);

const MessageSquareIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
    </svg>
);

const UserIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
    </svg>
);

const LogoutIcon = () => (
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

const SettingsIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="4" y1="6" x2="20" y2="6" /><line x1="8" y1="12" x2="20" y2="12" /><line x1="4" y1="18" x2="20" y2="18" />
        <circle cx="4" cy="6" r="2" fill="currentColor" stroke="none" /><circle cx="8" cy="12" r="2" fill="currentColor" stroke="none" /><circle cx="4" cy="18" r="2" fill="currentColor" stroke="none" />
    </svg>
);

const FaqIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>
);

const TimetableIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="4" width="18" height="18" rx="2" ry="2" /><line x1="16" y1="2" x2="16" y2="6" /><line x1="8" y1="2" x2="8" y2="6" /><line x1="3" y1="10" x2="21" y2="10" />
        <line x1="8" y1="14" x2="10" y2="14" /><line x1="14" y1="14" x2="16" y2="14" /><line x1="8" y1="18" x2="10" y2="18" /><line x1="14" y1="18" x2="16" y2="18" />
    </svg>
);

const adminSections = [
    { id: "overview", label: "Dashboard", detail: "KPIs and system health", icon: <DashboardIcon /> },
    { id: "upload", label: "Upload", detail: "Add knowledge files", icon: <UploadIcon /> },
    { id: "documents", label: "Documents", detail: "Manage indexed content", icon: <DocumentIcon /> },
    { id: "vector", label: "Vector DB", detail: "Index status", icon: <VectorIcon /> },
    { id: "faq", label: "FAQs", detail: "Manage Q&A", icon: <FaqIcon /> },
    { id: "timetable", label: "Timetable", detail: "Fetch UDOM timetables", icon: <TimetableIcon /> },
    { id: "feedback", label: "Feedback", detail: "User responses", icon: <MessageSquareIcon /> },
    { id: "settings", label: "RAG Settings", detail: "Pipeline configuration", icon: <SettingsIcon /> },
];

export default function AdminLayout({ children }) {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [currentSection, setCurrentSection] = useState(() => window.location.hash.replace("#", "") || "overview");
    const session = getAdminSession();

    // Password Change State
    const [isPasswordModalOpen, setIsPasswordModalOpen] = useState(false);
    const [pwdOld, setPwdOld] = useState("");
    const [pwdNew, setPwdNew] = useState("");
    const [pwdError, setPwdError] = useState("");
    const [pwdSuccess, setPwdSuccess] = useState("");
    const [isChangingPwd, setIsChangingPwd] = useState(false);

    useEffect(() => {
        const handleHashChange = () => {
            setCurrentSection(window.location.hash.replace("#", "") || "overview");
        };

        window.addEventListener("hashchange", handleHashChange);
        return () => window.removeEventListener("hashchange", handleHashChange);
    }, []);

    const handleLogout = () => {
        clearAdminSession();
        window.history.replaceState({}, "", "/login");
        window.dispatchEvent(new PopStateEvent("popstate"));
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
            await changePassword(session.id, pwdOld, pwdNew);
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

    return (
        <div className={`admin-shell ${isSidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
            <aside className="admin-sidebar">
                <button
                    type="button"
                    className="sidebar-toggle admin-sidebar-toggle"
                    onClick={() => setIsSidebarOpen((value) => !value)}
                    aria-label={isSidebarOpen ? "Close admin sidebar" : "Open admin sidebar"}
                    title={isSidebarOpen ? "Close sidebar" : "Open sidebar"}
                >
                    <span className="sidebar-toggle-icon" aria-hidden="true">
                        <span />
                        <span />
                    </span>
                </button>

                {isSidebarOpen && (
                    <div className="admin-sidebar-content">
                        <div className="admin-brand">
                            <span className="brand-mark">
                                <img src={udomLogo} alt="The University of Dodoma" />
                            </span>
                            <div>
                                <p className="admin-brand-eyebrow">Admin</p>
                                <h1>UDOM Control</h1>
                            </div>
                        </div>

                        <nav className="admin-side-nav" aria-label="Admin navigation">
                            {adminSections.map((section) => (
                                <a
                                    key={section.id}
                                    href={`/admin/dashboard#${section.id}`}
                                    className={currentSection === section.id ? "active" : ""}
                                >
                                    <span className="admin-nav-icon">{section.icon}</span>
                                    <span className="admin-nav-copy">
                                        <strong>{section.label}</strong>
                                        <small>{section.detail}</small>
                                    </span>
                                </a>
                            ))}
                        </nav>

                        <div className="admin-user-card">
                            <div className="admin-user-info">
                                <span className="admin-user-avatar"><UserIcon /></span>
                                <span>{session?.username || "Admin"}</span>
                            </div>
                            <div style={{ display: 'flex', gap: '8px' }}>
                                <button type="button" onClick={() => setIsPasswordModalOpen(true)} aria-label="Change Password" title="Change Password">
                                    <KeyIcon />
                                </button>
                                <button type="button" onClick={handleLogout} aria-label="Logout" title="Logout">
                                    <LogoutIcon />
                                </button>
                            </div>
                        </div>

                        <NavigationLink to="/" className="admin-public-link" onNavigate={clearAdminSession}>
                            Public site
                        </NavigationLink>
                    </div>
                )}
            </aside>

            <main className="admin-content">{children}</main>

            {/* Change Password Modal using the same styles as ChatWindow for simplicity */}
            {isPasswordModalOpen && (
                <div style={{
                    position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
                    backgroundColor: "rgba(0,0,0,0.5)", backdropFilter: "blur(4px)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    zIndex: 1000, padding: "20px"
                }}>
                    <div style={{
                        backgroundColor: "#131e2d", border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: "16px", padding: "24px", width: "100%", maxWidth: "400px",
                        boxShadow: "0 24px 48px rgba(0,0,0,0.4)"
                    }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "20px" }}>
                            <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#fff", margin: 0 }}>Change Password</h3>
                            <button type="button" style={{ background: "none", border: "none", color: "#8a95a5", cursor: "pointer", display: "flex", padding: "4px", borderRadius: "6px" }} onClick={() => setIsPasswordModalOpen(false)}>
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                                </svg>
                            </button>
                        </div>
                        <form onSubmit={handleChangePasswordSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                            <input
                                type="password"
                                placeholder="Current Password"
                                value={pwdOld}
                                onChange={(e) => setPwdOld(e.target.value)}
                                style={{ width: "100%", backgroundColor: "#0f1923", border: "1px solid rgba(255,255,255,0.1)", color: "#fff", padding: "12px 14px", borderRadius: "8px", fontSize: "14px", outline: "none" }}
                                required
                            />
                            <input
                                type="password"
                                placeholder="New Password (min 6 chars)"
                                value={pwdNew}
                                onChange={(e) => setPwdNew(e.target.value)}
                                style={{ width: "100%", backgroundColor: "#0f1923", border: "1px solid rgba(255,255,255,0.1)", color: "#fff", padding: "12px 14px", borderRadius: "8px", fontSize: "14px", outline: "none" }}
                                required
                            />
                            {pwdError && <p style={{ color: "#ff6b6b", fontSize: "13px", margin: 0 }}>{pwdError}</p>}
                            {pwdSuccess && <p style={{ color: "#51cf66", fontSize: "13px", margin: 0 }}>{pwdSuccess}</p>}
                            <div style={{ display: "flex", justifyContent: "flex-end", gap: "12px", marginTop: "8px" }}>
                                <button type="button" onClick={() => setIsPasswordModalOpen(false)} style={{ padding: "10px 16px", borderRadius: "8px", background: "transparent", color: "#fff", border: "1px solid rgba(255,255,255,0.2)", cursor: "pointer", fontSize: "14px", fontWeight: "500" }}>
                                    Cancel
                                </button>
                                <button type="submit" disabled={isChangingPwd} style={{ padding: "10px 16px", borderRadius: "8px", background: "#3b82f6", color: "#fff", border: "none", cursor: "pointer", fontSize: "14px", fontWeight: "500" }}>
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
