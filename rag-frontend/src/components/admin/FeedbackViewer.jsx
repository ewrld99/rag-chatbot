import { useEffect, useState } from "react";
import { getFeedbackMessages } from "../../api/chatApi.js";

const ThumbsUpIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
    </svg>
);

const ThumbsDownIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
    </svg>
);

export default function FeedbackViewer() {
    const [feedbackList, setFeedbackList] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState("");

    const loadFeedback = async () => {
        setIsLoading(true);
        setError("");
        try {
            const data = await getFeedbackMessages();
            setFeedbackList(data || []);
        } catch (err) {
            setError(err.message || "Failed to load feedback");
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        loadFeedback();
    }, []);

    return (
        <section className="admin-section">
            <div className="admin-section-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                    <h3>Message Feedback</h3>
                    <p className="admin-section-desc">Review thumbs up / down feedback on assistant responses.</p>
                </div>
                <button type="button" className="admin-btn" onClick={loadFeedback} disabled={isLoading}>
                    {isLoading ? "Refreshing..." : "Refresh"}
                </button>
            </div>

            {error && <div className="admin-error-banner">{error}</div>}

            <div className="admin-card">
                {isLoading ? (
                    <div className="admin-empty-state">
                        <p>Loading feedback...</p>
                    </div>
                ) : feedbackList.length === 0 ? (
                    <div className="admin-empty-state">
                        <p>No feedback has been submitted yet.</p>
                    </div>
                ) : (
                    <table className="admin-table">
                        <thead>
                            <tr>
                                <th style={{ width: "15%" }}>Date</th>
                                <th style={{ width: "10%" }}>User</th>
                                <th style={{ width: "10%" }}>Feedback</th>
                                <th>Message Content</th>
                            </tr>
                        </thead>
                        <tbody>
                            {feedbackList.map((item) => (
                                <tr key={item.id}>
                                    <td className="admin-td-subtle">
                                        {new Date(item.created_at).toLocaleDateString()}{" "}
                                        {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </td>
                                    <td className="admin-td-subtle">
                                        {item.username}
                                    </td>
                                    <td>
                                        {item.feedback === 1 ? (
                                            <div style={{ color: "#4a9080", display: "flex", alignItems: "center", gap: "6px" }}>
                                                <ThumbsUpIcon /> <span style={{ fontSize: "12px", fontWeight: "bold" }}>Helpful</span>
                                            </div>
                                        ) : item.feedback === -1 ? (
                                            <div style={{ color: "#e53e3e", display: "flex", alignItems: "center", gap: "6px" }}>
                                                <ThumbsDownIcon /> <span style={{ fontSize: "12px", fontWeight: "bold" }}>Not helpful</span>
                                            </div>
                                        ) : (
                                            <span style={{ color: "var(--app-muted)" }}>None</span>
                                        )}
                                    </td>
                                    <td style={{ whiteSpace: "pre-wrap", fontSize: "13px", lineHeight: "1.5" }}>
                                        <div style={{ maxHeight: "80px", overflowY: "auto", paddingRight: "8px" }}>
                                            {item.content}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </section>
    );
}
