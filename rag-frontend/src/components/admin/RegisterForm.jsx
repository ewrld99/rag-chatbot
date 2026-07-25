import { useState } from "react";
import udomLogo from "../../assets/udom-logo.svg";

const UserPlusIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <line x1="19" y1="8" x2="19" y2="14" />
        <line x1="16" y1="11" x2="22" y2="11" />
    </svg>
);

const KeyIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="7.5" cy="15.5" r="5.5" />
        <path d="m21 2-9.6 9.6" />
        <path d="m15.5 7.5 3 3L22 7l-3-3" />
    </svg>
);

const AlertIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
);

export default function RegisterForm({ onRegister, onLoginClick }) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [registrationNumber, setRegistrationNumber] = useState("");
    const [programme, setProgramme] = useState("");
    const [campus, setCampus] = useState("");
    const [admissionYear, setAdmissionYear] = useState("");
    const [error, setError] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [focused, setFocused] = useState(null);

    const handleSubmit = async (event) => {
        event.preventDefault();
        setError("");

        if (!username.trim() || !password.trim()) {
            setError("Please enter both username and password.");
            return;
        }

        setIsLoading(true);
        try {
            await onRegister(username, password, registrationNumber, programme, campus, admissionYear);
        } catch (registerError) {
            setError(registerError.message || "Could not create account. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div style={styles.wrapper}>
            <div style={styles.card}>
                <div style={styles.header}>
                    <div style={styles.iconBadge}>
                        <img src={udomLogo} alt="The University of Dodoma" style={styles.logo} />
                    </div>
                    <span style={styles.eyebrow}>UDOM Chatbot</span>
                    <h2 style={styles.title}>Create account</h2>
                </div>

                <form style={styles.form} onSubmit={handleSubmit}>
                    <div style={styles.fields}>
                        <div style={styles.fieldGroup}>
                            <label style={styles.label} htmlFor="register-username">Username</label>
                            <div style={{
                                ...styles.inputWrapper,
                                ...(focused === "username" ? styles.inputWrapperFocused : {}),
                            }}>
                                <span style={styles.inputIcon}><UserPlusIcon /></span>
                                <input
                                    id="register-username"
                                    style={styles.input}
                                    value={username}
                                    onChange={(e) => { setUsername(e.target.value); setError(""); }}
                                    onFocus={() => setFocused("username")}
                                    onBlur={() => setFocused(null)}
                                    autoComplete="username"
                                    placeholder="Username"
                                    required
                                />
                            </div>
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label} htmlFor="register-reg">Registration Number</label>
                            <div style={{
                                ...styles.inputWrapper,
                                ...(focused === "reg" ? styles.inputWrapperFocused : {}),
                            }}>
                                <span style={styles.inputIcon}><UserPlusIcon /></span>
                                <input
                                    id="register-reg"
                                    style={styles.input}
                                    value={registrationNumber}
                                    onChange={(e) => { setRegistrationNumber(e.target.value); setError(""); }}
                                    onFocus={() => setFocused("reg")}
                                    onBlur={() => setFocused(null)}
                                    placeholder="e.g. T21-03-0000"
                                />
                            </div>
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label} htmlFor="register-prog">Programme</label>
                            <div style={{
                                ...styles.inputWrapper,
                                ...(focused === "prog" ? styles.inputWrapperFocused : {}),
                            }}>
                                <span style={styles.inputIcon}><UserPlusIcon /></span>
                                <input
                                    id="register-prog"
                                    style={styles.input}
                                    value={programme}
                                    onChange={(e) => { setProgramme(e.target.value); setError(""); }}
                                    onFocus={() => setFocused("prog")}
                                    onBlur={() => setFocused(null)}
                                    placeholder="e.g. Computer Science"
                                />
                            </div>
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label} htmlFor="register-campus">Campus</label>
                            <div style={{
                                ...styles.inputWrapper,
                                ...(focused === "campus" ? styles.inputWrapperFocused : {}),
                            }}>
                                <span style={styles.inputIcon}><UserPlusIcon /></span>
                                <input
                                    id="register-campus"
                                    style={styles.input}
                                    value={campus}
                                    onChange={(e) => { setCampus(e.target.value); setError(""); }}
                                    onFocus={() => setFocused("campus")}
                                    onBlur={() => setFocused(null)}
                                    placeholder="e.g. CIVE"
                                />
                            </div>
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label} htmlFor="register-year">Admission Year</label>
                            <div style={{
                                ...styles.inputWrapper,
                                ...(focused === "year" ? styles.inputWrapperFocused : {}),
                            }}>
                                <span style={styles.inputIcon}><UserPlusIcon /></span>
                                <input
                                    id="register-year"
                                    style={styles.input}
                                    type="number"
                                    value={admissionYear}
                                    onChange={(e) => { setAdmissionYear(e.target.value); setError(""); }}
                                    onFocus={() => setFocused("year")}
                                    onBlur={() => setFocused(null)}
                                    placeholder="e.g. 2025"
                                />
                            </div>
                        </div>

                        <div style={styles.fieldGroup}>
                            <label style={styles.label} htmlFor="register-password">Password</label>
                            <div style={{
                                ...styles.inputWrapper,
                                ...(focused === "password" ? styles.inputWrapperFocused : {}),
                            }}>
                                <span style={styles.inputIcon}><KeyIcon /></span>
                                <input
                                    id="register-password"
                                    style={styles.input}
                                    type="password"
                                    value={password}
                                    onChange={(e) => { setPassword(e.target.value); setError(""); }}
                                    onFocus={() => setFocused("password")}
                                    onBlur={() => setFocused(null)}
                                    autoComplete="new-password"
                                    placeholder="Create password"
                                    required
                                />
                            </div>
                        </div>
                    </div>

                    {error && (
                        <div style={styles.errorBox} role="alert">
                            <AlertIcon />
                            <span>{error}</span>
                        </div>
                    )}

                    <button
                        type="submit"
                        style={{ ...styles.button, ...(isLoading ? styles.buttonDisabled : {}) }}
                        disabled={isLoading}
                    >
                        <span style={styles.buttonInner}>
                            {isLoading ? <span style={styles.spinner} /> : <UserPlusIcon />}
                            {isLoading ? "Creating account..." : "Create account"}
                        </span>
                    </button>

                    <button type="button" style={styles.linkButton} onClick={onLoginClick}>
                        Already have an account?
                    </button>
                </form>
            </div>
        </div>
    );
}

const styles = {
    wrapper: {
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
    },
    card: {
        width: "100%",
        maxWidth: "420px",
        background: "var(--app-surface)",
        border: "1px solid var(--app-border)",
        borderRadius: "18px",
        padding: "34px",
        boxShadow: "0 24px 70px rgba(0, 0, 0, 0.16)",
    },
    header: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        gap: "10px",
        marginBottom: "28px",
    },
    iconBadge: {
        width: "78px",
        height: "78px",
        borderRadius: "50%",
        background: "var(--app-surface-muted)",
        border: "1px solid var(--app-border-strong)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: "2px",
        overflow: "hidden",
    },
    logo: {
        width: "100%",
        height: "100%",
        objectFit: "cover",
        display: "block",
    },
    eyebrow: {
        fontSize: "11px",
        fontWeight: "700",
        letterSpacing: "1.2px",
        textTransform: "uppercase",
        color: "var(--app-accent)",
    },
    title: {
        fontSize: "28px",
        fontWeight: "700",
        color: "var(--app-text)",
        margin: 0,
        letterSpacing: 0,
    },
    form: {
        display: "flex",
        flexDirection: "column",
        gap: "18px",
    },
    fields: {
        display: "flex",
        flexDirection: "column",
        gap: "16px",
    },
    fieldGroup: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
    },
    label: {
        fontSize: "13px",
        fontWeight: "600",
        color: "var(--app-text)",
        letterSpacing: "0.2px",
    },
    inputWrapper: {
        display: "flex",
        alignItems: "center",
        background: "var(--app-bg)",
        border: "1px solid var(--app-border-strong)",
        borderRadius: "12px",
        transition: "border-color 0.2s, box-shadow 0.2s",
        overflow: "hidden",
    },
    inputWrapperFocused: {
        borderColor: "var(--app-accent-strong)",
        boxShadow: "0 0 0 3px rgba(217,108,71,0.12)",
    },
    inputIcon: {
        display: "flex",
        alignItems: "center",
        paddingLeft: "14px",
        color: "var(--app-faint)",
        flexShrink: 0,
    },
    input: {
        flex: 1,
        background: "transparent",
        border: "none",
        outline: "none",
        padding: "13px 14px",
        fontSize: "14px",
        color: "var(--app-text)",
        width: "100%",
    },
    errorBox: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "12px 14px",
        borderRadius: "10px",
        background: "var(--app-danger-soft)",
        border: "1px solid var(--app-border-strong)",
        color: "var(--app-danger)",
        fontSize: "13px",
    },
    button: {
        width: "100%",
        padding: "14px",
        borderRadius: "12px",
        border: "none",
        background: "var(--app-text)",
        color: "var(--app-surface-muted)",
        fontSize: "14px",
        fontWeight: "700",
        cursor: "pointer",
        transition: "opacity 0.2s, transform 0.1s",
        letterSpacing: "0.2px",
    },
    buttonDisabled: {
        opacity: 0.7,
        cursor: "not-allowed",
    },
    buttonInner: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "8px",
    },
    linkButton: {
        border: "none",
        background: "transparent",
        color: "var(--app-accent)",
        fontSize: "13px",
        fontWeight: "700",
        cursor: "pointer",
        padding: 0,
    },
    spinner: {
        width: "16px",
        height: "16px",
        border: "2px solid var(--app-border-strong)",
        borderTop: "2px solid var(--app-surface-muted)",
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
        display: "inline-block",
    },
};
