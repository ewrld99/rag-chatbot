import { useEffect, useState } from "react";
import { fetchFaqs, createFaq, updateFaq, deleteFaq, importFaqs } from "../../api/faqApi.js";

const EditIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" /><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" /></svg>
);
const DeleteIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
);
const UploadIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
);
const PlusIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
);

export default function FaqManager() {
    const [faqs, setFaqs] = useState([]);
    const [categories, setCategories] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState("");
    const [filterCategory, setFilterCategory] = useState("");
    
    // Modal state
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editingFaq, setEditingFaq] = useState(null);
    const [formData, setFormData] = useState({ question: "", answer: "", category: "", is_active: true });
    
    // Import state
    const [isImporting, setIsImporting] = useState(false);
    const [importMsg, setImportMsg] = useState("");

    const loadData = async () => {
        setIsLoading(true);
        try {
            const data = await fetchFaqs({ search: searchTerm, category: filterCategory });
            setFaqs(data.items);
            setCategories(data.categories.filter(Boolean));
        } catch (error) {
            console.error("Failed to load FAQs:", error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, [searchTerm, filterCategory]);

    const handleSave = async (e) => {
        e.preventDefault();
        try {
            if (editingFaq) {
                await updateFaq(editingFaq.id, formData);
            } else {
                await createFaq(formData);
            }
            setIsModalOpen(false);
            loadData();
        } catch (error) {
            alert(error.message);
        }
    };

    const handleDelete = async (id) => {
        if (!confirm("Are you sure you want to delete this FAQ?")) return;
        try {
            await deleteFaq(id);
            loadData();
        } catch (error) {
            alert(error.message);
        }
    };

    const handleToggleActive = async (faq) => {
        try {
            await updateFaq(faq.id, { is_active: !faq.is_active });
            loadData();
        } catch (error) {
            alert(error.message);
        }
    };

    const handleImport = async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        
        setIsImporting(true);
        setImportMsg("Importing...");
        try {
            const stats = await importFaqs(file);
            setImportMsg(`Imported: ${stats.imported}, Skipped: ${stats.skipped}, Duplicates: ${stats.duplicates}, Errors: ${stats.errors}`);
            loadData();
        } catch (error) {
            setImportMsg("Import failed: " + error.message);
        } finally {
            setIsImporting(false);
            e.target.value = "";
        }
    };

    const openModal = (faq = null) => {
        if (faq) {
            setEditingFaq(faq);
            setFormData({ question: faq.question, answer: faq.answer, category: faq.category || "", is_active: faq.is_active });
        } else {
            setEditingFaq(null);
            setFormData({ question: "", answer: "", category: "", is_active: true });
        }
        setIsModalOpen(true);
    };

    return (
        <section className="dashboard-card" style={{ padding: "28px", borderRadius: "20px", background: "var(--app-surface)", border: "1px solid var(--app-border)", display: "flex", flexDirection: "column", gap: "20px" }}>
            <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
                <div>
                    <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--app-accent)" }}>Management</span>
                    <h2 style={{ fontSize: "20px", fontWeight: 700, margin: 0, color: "var(--app-text)" }}>FAQs</h2>
                </div>
                <div style={{ display: "flex", gap: "10px" }}>
                    <label style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 14px", borderRadius: "10px", background: "var(--app-bg)", border: "1px solid var(--app-border)", cursor: "pointer", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                        <UploadIcon />
                        {isImporting ? "Importing..." : "Bulk Import"}
                        <input type="file" accept=".csv,.xlsx,.xls" style={{ display: "none" }} onChange={handleImport} disabled={isImporting} />
                    </label>
                    <button onClick={() => openModal()} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 14px", borderRadius: "10px", background: "var(--app-text)", color: "var(--app-surface-muted)", border: "none", cursor: "pointer", fontSize: "13px", fontWeight: 600 }}>
                        <PlusIcon /> Add FAQ
                    </button>
                </div>
            </header>

            {importMsg && <div style={{ padding: "12px", background: "var(--app-accent-soft)", color: "var(--app-accent)", borderRadius: "8px", fontSize: "13px" }}>{importMsg}</div>}

            <div style={{ display: "flex", gap: "12px" }}>
                <input 
                    type="text" 
                    placeholder="Search FAQs..." 
                    value={searchTerm} 
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={{ flex: 1, padding: "10px 14px", borderRadius: "10px", border: "1px solid var(--app-border)", background: "var(--app-surface-muted)", outline: "none", fontSize: "14px" }}
                />
                <select 
                    value={filterCategory} 
                    onChange={(e) => setFilterCategory(e.target.value)}
                    style={{ padding: "10px 14px", borderRadius: "10px", border: "1px solid var(--app-border)", background: "var(--app-surface-muted)", outline: "none", fontSize: "14px" }}
                >
                    <option value="">All Categories</option>
                    {categories.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
            </div>

            <div style={{ background: "#ffffff", border: "1px solid var(--app-border)", borderRadius: "14px", overflow: "hidden" }}>
                {isLoading ? (
                    <div style={{ padding: "40px", textAlign: "center", color: "var(--app-faint)" }}>Loading...</div>
                ) : faqs.length === 0 ? (
                    <div style={{ padding: "40px", textAlign: "center", color: "var(--app-faint)" }}>No FAQs found.</div>
                ) : (
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
                        <thead>
                            <tr style={{ background: "#f7f5f0", borderBottom: "1px solid var(--app-border)" }}>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)" }}>Question</th>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)" }}>Category</th>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)" }}>Status</th>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)", textAlign: "right" }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {faqs.map(faq => (
                                <tr key={faq.id} style={{ borderBottom: "1px solid var(--app-bg)" }}>
                                    <td style={{ padding: "14px 16px", color: "var(--app-text)", maxWidth: "300px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{faq.question}</td>
                                    <td style={{ padding: "14px 16px", color: "#4d4942" }}>{faq.category || "—"}</td>
                                    <td style={{ padding: "14px 16px" }}>
                                        <button onClick={() => handleToggleActive(faq)} style={{ padding: "4px 8px", borderRadius: "20px", border: "none", fontSize: "11px", fontWeight: 700, cursor: "pointer", background: faq.is_active ? "#ecfdf5" : "#fef2f2", color: faq.is_active ? "#0f766e" : "#b91c1c" }}>
                                            {faq.is_active ? "Active" : "Disabled"}
                                        </button>
                                    </td>
                                    <td style={{ padding: "14px 16px", textAlign: "right" }}>
                                        <button onClick={() => openModal(faq)} style={{ background: "none", border: "none", color: "var(--app-faint)", cursor: "pointer", padding: "4px" }}><EditIcon /></button>
                                        <button onClick={() => handleDelete(faq.id)} style={{ background: "none", border: "none", color: "var(--app-accent-strong)", cursor: "pointer", padding: "4px", marginLeft: "4px" }}><DeleteIcon /></button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Modal */}
            {isModalOpen && (
                <div style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, background: "rgba(43, 41, 37, 0.4)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 }}>
                    <div style={{ background: "var(--app-surface)", borderRadius: "16px", width: "100%", maxWidth: "500px", padding: "24px", border: "1px solid var(--app-border)", boxShadow: "0 24px 48px rgba(72, 61, 47, 0.2)" }}>
                        <h3 style={{ margin: "0 0 16px", color: "var(--app-text)" }}>{editingFaq ? "Edit FAQ" : "Add FAQ"}</h3>
                        <form onSubmit={handleSave} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                            <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                Question *
                                <textarea required value={formData.question} onChange={e => setFormData({...formData, question: e.target.value})} maxLength={500} style={{ padding: "10px", borderRadius: "8px", border: "1px solid var(--app-border)", minHeight: "60px", resize: "vertical" }} />
                            </label>
                            <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                Answer *
                                <textarea required value={formData.answer} onChange={e => setFormData({...formData, answer: e.target.value})} maxLength={10000} style={{ padding: "10px", borderRadius: "8px", border: "1px solid var(--app-border)", minHeight: "120px", resize: "vertical" }} />
                            </label>
                            <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                Category
                                <input type="text" value={formData.category} onChange={e => setFormData({...formData, category: e.target.value})} placeholder="e.g., Admissions" style={{ padding: "10px", borderRadius: "8px", border: "1px solid var(--app-border)" }} />
                            </label>
                            <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                <input type="checkbox" checked={formData.is_active} onChange={e => setFormData({...formData, is_active: e.target.checked})} />
                                Is Active
                            </label>
                            <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", marginTop: "10px" }}>
                                <button type="button" onClick={() => setIsModalOpen(false)} style={{ padding: "8px 16px", borderRadius: "8px", background: "var(--app-bg)", border: "1px solid var(--app-border)", color: "#4d4942", cursor: "pointer", fontWeight: 600 }}>Cancel</button>
                                <button type="submit" style={{ padding: "8px 16px", borderRadius: "8px", background: "var(--app-text)", border: "none", color: "var(--app-surface-muted)", cursor: "pointer", fontWeight: 600 }}>Save</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </section>
    );
}
