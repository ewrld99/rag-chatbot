import { useEffect, useState, useMemo } from "react";
import Dashboard from "../components/admin/Dashboard.jsx";
import DocumentManager from "../components/admin/DocumentManager.jsx";
import UploadCard from "../components/admin/UploadCard.jsx";
import VectorStatus from "../components/admin/VectorStatus.jsx";
import FeedbackViewer from "../components/admin/FeedbackViewer.jsx";
import RagSettings from "../components/admin/RagSettings.jsx";
import FaqManager from "../components/admin/FaqManager.jsx";
import TimetableFetcher from "../components/admin/TimetableFetcher.jsx";
import { getDocuments } from "../api/documentApi.js";

function getCurrentSection() {
    return window.location.hash.replace("#", "") || "overview";
}

export default function AdminDashboardPage() {
    const [refreshKey, setRefreshKey] = useState(0);
    const [section, setSection] = useState(getCurrentSection);
    const [documents, setDocuments] = useState([]);
    const [status, setStatus] = useState({ isLoading: true, error: "" });

    useEffect(() => {
        const handleHashChange = () => setSection(getCurrentSection());
        window.addEventListener("hashchange", handleHashChange);

        return () => window.removeEventListener("hashchange", handleHashChange);
    }, []);

    useEffect(() => {
        let isActive = true;

        async function loadStats() {
            setStatus({ isLoading: true, error: "" });

            try {
                const data = await getDocuments();
                if (isActive) {
                    setDocuments(data);
                    setStatus({ isLoading: false, error: "" });
                }
            } catch (error) {
                if (isActive) {
                    setStatus({ isLoading: false, error: error.message });
                }
            }
        }

        loadStats();

        return () => {
            isActive = false;
        };
    }, [refreshKey]);

    const stats = useMemo(() => ({
        documents: documents.length,
        chunks: documents.reduce((total, document) => total + (document.chunk_count ?? 1), 0),
        sources: new Set(documents.map((document) => document.filename).filter(Boolean)).size,
        isLoading: status.isLoading,
        error: status.error,
    }), [documents, status.isLoading, status.error]);

    return (
        <div className="admin-dashboard-page">
            <header className="admin-page-header">
                <div>
                    <h2>Dashboard</h2>
                </div>
                <span className={`system-pill ${status.error ? "warning" : "healthy"}`}>
                    {status.error ? "Needs attention" : "System ready"}
                </span>
            </header>

            {section === "overview" && <Dashboard stats={stats} />}
            {section === "upload" && <UploadCard onUploaded={() => setRefreshKey((value) => value + 1)} />}
            {section === "documents" && <DocumentManager refreshKey={refreshKey} onChanged={() => setRefreshKey((value) => value + 1)} />}
            {section === "vector" && <VectorStatus stats={stats} documents={documents} />}
            {section === "faq" && <FaqManager />}
            {section === "timetable" && <TimetableFetcher />}
            {section === "feedback" && <FeedbackViewer />}
            {section === "settings" && <RagSettings />}
        </div>
    );
}
