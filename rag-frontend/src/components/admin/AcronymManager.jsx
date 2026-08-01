import { useEffect, useMemo, useState } from "react";
import { createAlias, deleteAlias, fetchAliases, importAliases, updateAlias } from "../../api/aliasApi.js";

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

const emptyForm = { term: "", aliasesText: "", category: "", weight: 1, is_active: true };

function aliasesToText(aliases = []) {
    return aliases.join("; ");
}

function parseAliases(value) {
    return value
        .split(/[;\n,]/)
        .map((item) => item.trim().toLowerCase())
        .filter(Boolean)
        .filter((item, index, items) => items.indexOf(item) === index);
}

export default function AcronymManager() {
    const [aliases, setAliases] = useState([]);
    const [categories, setCategories] = useState([]);
    const [searchTerm, setSearchTerm] = useState("");
    const [filterCategory, setFilterCategory] = useState("");
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState("");
    const [notice, setNotice] = useState("");
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editingAlias, setEditingAlias] = useState(null);
    const [formData, setFormData] = useState(emptyForm);
    const [isSaving, setIsSaving] = useState(false);
    const [isImporting, setIsImporting] = useState(false);

    async function loadData() {
        setIsLoading(true);
        setError("");
        try {
            const data = await fetchAliases({ search: searchTerm, category: filterCategory });
            setAliases(data.items || []);
            setCategories((data.categories || []).filter(Boolean));
        } catch (loadError) {
            setError(loadError.message || "Failed to load acronyms.");
        } finally {
            setIsLoading(false);
        }
    }

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        loadData();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [searchTerm, filterCategory]);

    const totalAliasTerms = useMemo(
        () => aliases.reduce((total, item) => total + 1 + (item.aliases?.length || 0), 0),
        [aliases],
    );

    function openModal(alias = null) {
        setNotice("");
        setError("");
        setEditingAlias(alias);
        setFormData(alias ? {
            term: alias.term,
            aliasesText: aliasesToText(alias.aliases),
            category: alias.category || "",
            weight: alias.weight || 1,
            is_active: alias.is_active,
        } : emptyForm);
        setIsModalOpen(true);
    }

    async function handleSave(event) {
        event.preventDefault();
        setIsSaving(true);
        setError("");
        const payload = {
            term: formData.term.trim().toLowerCase(),
            aliases: parseAliases(formData.aliasesText),
            category: formData.category.trim().toLowerCase() || null,
            weight: Number(formData.weight) || 1,
            is_active: formData.is_active,
        };

        try {
            if (editingAlias) {
                await updateAlias(editingAlias.id, payload);
                setNotice("Acronym updated.");
            } else {
                await createAlias(payload);
                setNotice("Acronym added.");
            }
            setIsModalOpen(false);
            await loadData();
        } catch (saveError) {
            setError(saveError.message || "Failed to save acronym.");
        } finally {
            setIsSaving(false);
        }
    }

    async function handleToggle(alias) {
        try {
            await updateAlias(alias.id, { is_active: !alias.is_active });
            await loadData();
        } catch (toggleError) {
            setError(toggleError.message || "Failed to update status.");
        }
    }

    async function handleDelete(alias) {
        if (!confirm(`Delete ${alias.term}?`)) return;
        try {
            await deleteAlias(alias.id);
            setNotice("Acronym deleted.");
            await loadData();
        } catch (deleteError) {
            setError(deleteError.message || "Failed to delete acronym.");
        }
    }

    async function handleImport(event) {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsImporting(true);
        setNotice("Importing CSV...");
        setError("");
        try {
            const result = await importAliases(file);
            setNotice(`Created ${result.created}, updated ${result.updated}, skipped ${result.skipped}.`);
            if (result.errors?.length) {
                setError(result.errors.join(" "));
            }
            await loadData();
        } catch (importError) {
            setError(importError.message || "Failed to import CSV.");
            setNotice("");
        } finally {
            setIsImporting(false);
            event.target.value = "";
        }
    }

    return (
        <section className="dashboard-card" style={{ padding: "28px", borderRadius: "20px", background: "var(--app-surface)", border: "1px solid var(--app-border)", display: "flex", flexDirection: "column", gap: "20px" }}>
            <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: "16px", flexWrap: "wrap" }}>
                <div>
                    <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "1.5px", textTransform: "uppercase", color: "var(--app-accent)" }}>Retrieval vocabulary</span>
                    <h2 style={{ fontSize: "20px", fontWeight: 700, margin: 0, color: "var(--app-text)" }}>Acronyms</h2>
                    <p style={{ margin: "6px 0 0", color: "var(--app-muted)", fontSize: "13px" }}>{aliases.length} terms, {totalAliasTerms} searchable words and phrases</p>
                    <p style={{ margin: "4px 0 0", color: "var(--app-faint)", fontSize: "12px" }}>CSV headers: term/acronym and aliases/full form/meaning.</p>
                </div>
                <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                    <label style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 14px", borderRadius: "10px", background: "var(--app-bg)", border: "1px solid var(--app-border)", cursor: "pointer", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                        <UploadIcon />
                        {isImporting ? "Importing..." : "Import CSV"}
                        <input type="file" accept=".csv" style={{ display: "none" }} onChange={handleImport} disabled={isImporting} />
                    </label>
                    <button type="button" onClick={() => openModal()} style={{ display: "flex", alignItems: "center", gap: "6px", padding: "8px 14px", borderRadius: "10px", background: "var(--app-text)", color: "var(--app-surface-muted)", border: "none", cursor: "pointer", fontSize: "13px", fontWeight: 600 }}>
                        <PlusIcon /> Add Acronym
                    </button>
                </div>
            </header>

            {notice && <div style={{ padding: "12px", background: "var(--app-accent-soft)", color: "var(--app-accent)", borderRadius: "8px", fontSize: "13px" }}>{notice}</div>}
            {error && <div style={{ padding: "12px", background: "#fef2f2", color: "#b91c1c", borderRadius: "8px", fontSize: "13px" }}>{error}</div>}

            <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
                <input
                    type="text"
                    placeholder="Search acronyms or aliases..."
                    value={searchTerm}
                    onChange={(event) => setSearchTerm(event.target.value)}
                    style={{ flex: "1 1 260px", padding: "10px 14px", borderRadius: "10px", border: "1px solid var(--app-border)", background: "var(--app-surface-muted)", outline: "none", fontSize: "14px" }}
                />
                <select
                    value={filterCategory}
                    onChange={(event) => setFilterCategory(event.target.value)}
                    style={{ padding: "10px 14px", borderRadius: "10px", border: "1px solid var(--app-border)", background: "var(--app-surface-muted)", outline: "none", fontSize: "14px" }}
                >
                    <option value="">All Categories</option>
                    {categories.map((category) => <option key={category} value={category}>{category}</option>)}
                </select>
            </div>

            <div style={{ background: "#ffffff", border: "1px solid var(--app-border)", borderRadius: "14px", overflow: "hidden" }}>
                {isLoading ? (
                    <div style={{ padding: "40px", textAlign: "center", color: "var(--app-faint)" }}>Loading...</div>
                ) : aliases.length === 0 ? (
                    <div style={{ padding: "40px", textAlign: "center", color: "var(--app-faint)" }}>No acronyms found.</div>
                ) : (
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", textAlign: "left" }}>
                        <thead>
                            <tr style={{ background: "#f7f5f0", borderBottom: "1px solid var(--app-border)" }}>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)" }}>Term</th>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)" }}>Aliases</th>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)" }}>Category</th>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)" }}>Status</th>
                                <th style={{ padding: "12px 16px", fontWeight: 600, color: "var(--app-muted)", textAlign: "right" }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {aliases.map((alias) => (
                                <tr key={alias.id} style={{ borderBottom: "1px solid var(--app-bg)" }}>
                                    <td style={{ padding: "14px 16px", color: "var(--app-text)", fontWeight: 700, textTransform: "uppercase" }}>{alias.term}</td>
                                    <td style={{ padding: "14px 16px", color: "#4d4942", maxWidth: "520px" }}>{aliasesToText(alias.aliases) || "None"}</td>
                                    <td style={{ padding: "14px 16px", color: "#4d4942" }}>{alias.category || "-"}</td>
                                    <td style={{ padding: "14px 16px" }}>
                                        <button type="button" onClick={() => handleToggle(alias)} style={{ padding: "4px 8px", borderRadius: "20px", border: "none", fontSize: "11px", fontWeight: 700, cursor: "pointer", background: alias.is_active ? "#ecfdf5" : "#fef2f2", color: alias.is_active ? "#0f766e" : "#b91c1c" }}>
                                            {alias.is_active ? "Active" : "Disabled"}
                                        </button>
                                    </td>
                                    <td style={{ padding: "14px 16px", textAlign: "right", whiteSpace: "nowrap" }}>
                                        <button type="button" onClick={() => openModal(alias)} style={{ background: "none", border: "none", color: "var(--app-faint)", cursor: "pointer", padding: "4px" }} aria-label={`Edit ${alias.term}`}><EditIcon /></button>
                                        <button type="button" onClick={() => handleDelete(alias)} style={{ background: "none", border: "none", color: "var(--app-accent-strong)", cursor: "pointer", padding: "4px", marginLeft: "4px" }} aria-label={`Delete ${alias.term}`}><DeleteIcon /></button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {isModalOpen && (
                <div style={{ position: "fixed", inset: 0, background: "rgba(43, 41, 37, 0.4)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100, padding: "20px" }}>
                    <div style={{ background: "var(--app-surface)", borderRadius: "16px", width: "100%", maxWidth: "560px", padding: "24px", border: "1px solid var(--app-border)", boxShadow: "0 24px 48px rgba(72, 61, 47, 0.2)" }}>
                        <h3 style={{ margin: "0 0 16px", color: "var(--app-text)" }}>{editingAlias ? "Edit Acronym" : "Add Acronym"}</h3>
                        <form onSubmit={handleSave} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                            <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                Term
                                <input required value={formData.term} onChange={(event) => setFormData({ ...formData, term: event.target.value })} placeholder="e.g., sr" maxLength={120} style={{ padding: "10px", borderRadius: "8px", border: "1px solid var(--app-border)" }} />
                            </label>
                            <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                Aliases
                                <textarea value={formData.aliasesText} onChange={(event) => setFormData({ ...formData, aliasesText: event.target.value })} placeholder="Separate aliases with semicolons, commas, or new lines" style={{ padding: "10px", borderRadius: "8px", border: "1px solid var(--app-border)", minHeight: "100px", resize: "vertical" }} />
                            </label>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 120px", gap: "12px" }}>
                                <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                    Category
                                    <input value={formData.category} onChange={(event) => setFormData({ ...formData, category: event.target.value })} placeholder="e.g., portal" maxLength={120} style={{ padding: "10px", borderRadius: "8px", border: "1px solid var(--app-border)" }} />
                                </label>
                                <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                    Weight
                                    <input type="number" min="0.1" max="5" step="0.1" value={formData.weight} onChange={(event) => setFormData({ ...formData, weight: event.target.value })} style={{ padding: "10px", borderRadius: "8px", border: "1px solid var(--app-border)" }} />
                                </label>
                            </div>
                            <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "13px", fontWeight: 600, color: "#4d4942" }}>
                                <input type="checkbox" checked={formData.is_active} onChange={(event) => setFormData({ ...formData, is_active: event.target.checked })} />
                                Active
                            </label>
                            <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", marginTop: "10px" }}>
                                <button type="button" onClick={() => setIsModalOpen(false)} style={{ padding: "8px 16px", borderRadius: "8px", background: "var(--app-bg)", border: "1px solid var(--app-border)", color: "#4d4942", cursor: "pointer", fontWeight: 600 }}>Cancel</button>
                                <button type="submit" disabled={isSaving} style={{ padding: "8px 16px", borderRadius: "8px", background: "var(--app-text)", border: "none", color: "var(--app-surface-muted)", cursor: "pointer", fontWeight: 600 }}>{isSaving ? "Saving..." : "Save"}</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </section>
    );
}
