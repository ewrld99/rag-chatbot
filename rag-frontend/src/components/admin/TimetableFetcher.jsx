import { useState, useEffect, useCallback } from "react";
import {
    getYears,
    getSemesters,
    getCategories,
    getOptionTypes,
    getDataOptions,
    fetchTimetable,
    getFetchProgress,
} from "../../api/timetableApi";

/* ----------  SVG Icons  ---------- */
const CalendarIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="4" width="18" height="18" rx="2" ry="2" /><line x1="16" y1="2" x2="16" y2="6" /><line x1="8" y1="2" x2="8" y2="6" /><line x1="3" y1="10" x2="21" y2="10" />
    </svg>
);

const DownloadIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
    </svg>
);

const CheckIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12" />
    </svg>
);

const SpinnerIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="spin-icon">
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
);

const AlertIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
);

const OPTION_TYPES = [
    { value: "programme", label: "By Programme" },
    { value: "course", label: "By Course" },
    { value: "room", label: "By Room/Venue" },
    { value: "instructor", label: "By Instructor" },
];

const STRATEGIES = [
    { value: "timetable", label: "Timetable (Recommended)", desc: "Converts each row into a full sentence — best for grid timetable PDFs" },
    { value: "auto", label: "Auto", desc: "General purpose — good for mixed text and simple table PDFs" },
    { value: "fast", label: "Fast", desc: "Raw text only — fastest but ignores table structure" },
];

const INITIAL_FORM = {
    year: "",
    semester: "",
    category: "",
    option: "",
    data: [],
    label: "",
    strategy: "timetable",
};

export default function TimetableFetcher() {
    const [form, setForm] = useState(INITIAL_FORM);
    const [years, setYears] = useState([]);
    const [semesters, setSemesters] = useState([]);
    const [categories, setCategories] = useState([]);
    const [optionTypes, setOptionTypes] = useState([]);
    const [dataOptions, setDataOptions] = useState([]);

    const [loading, setLoading] = useState({ years: true, semesters: false, categories: false, optionTypes: false, dataOptions: false });
    const [status, setStatus] = useState({ type: "", message: "" }); // type: success | error | processing
    const [taskId, setTaskId] = useState(null);
    const [progressPct, setProgressPct] = useState(0);
    const [fetchHistory, setFetchHistory] = useState(() => {
        try { return JSON.parse(localStorage.getItem("tt_history") || "[]"); }
        catch { return []; }
    });

    /* ---- Load years on mount ---- */
    useEffect(() => {
        getYears()
            .then(setYears)
            .catch(() => setYears([{ value: "11", label: "2024/2025" }, { value: "12", label: "2025/2026" }]))
            .finally(() => setLoading(l => ({ ...l, years: false })));
    }, []);

    /* ---- Cascade: year → semesters ---- */
    useEffect(() => {
        if (!form.year) return;
        setSemesters([]);
        setCategories([]);
        setOptionTypes([]);
        setDataOptions([]);
        setForm(f => ({ ...f, semester: "", category: "", option: "", data: [] }));
        setLoading(l => ({ ...l, semesters: true }));

        getSemesters(form.year)
            .then(setSemesters)
            .catch(() => setSemesters([]))
            .finally(() => setLoading(l => ({ ...l, semesters: false })));
    }, [form.year]);

    /* ---- Cascade: semester → categories ---- */
    useEffect(() => {
        if (!form.semester) return;
        setCategories([]);
        setOptionTypes([]);
        setDataOptions([]);
        setForm(f => ({ ...f, category: "", option: "", data: [] }));
        setLoading(l => ({ ...l, categories: true }));

        getCategories(form.year, form.semester)
            .then(setCategories)
            .catch(() => setCategories([]))
            .finally(() => setLoading(l => ({ ...l, categories: false })));
    }, [form.semester]);

    /* ---- Cascade: category → option types ---- */
    useEffect(() => {
        if (!form.category) return;
        setOptionTypes([]);
        setDataOptions([]);
        setForm(f => ({ ...f, option: "", data: [] }));
        setLoading(l => ({ ...l, optionTypes: true }));

        getOptionTypes(form.year, form.semester, form.category)
            .then(setOptionTypes)
            .catch(() => setOptionTypes([]))
            .finally(() => setLoading(l => ({ ...l, optionTypes: false })));
    }, [form.category]);

    /* ---- Cascade: option → data list ---- */
    useEffect(() => {
        if (!form.option) return;
        setDataOptions([]);
        setForm(f => ({ ...f, data: [] }));
        setLoading(l => ({ ...l, dataOptions: true }));

        getDataOptions(form.year, form.semester, form.category, form.option)
            .then(setDataOptions)
            .catch(() => setDataOptions([]))
            .finally(() => setLoading(l => ({ ...l, dataOptions: false })));
    }, [form.option]);

    /* ---- Progress Polling ---- */
    useEffect(() => {
        if (!taskId) return;
        
        let interval = setInterval(async () => {
            try {
                const res = await getFetchProgress(taskId);
                if (res.status === "error") {
                    setStatus({ type: "error", message: res.progress?.message || "An error occurred." });
                    setTaskId(null);
                    clearInterval(interval);
                } else if (res.status === "success") {
                    setStatus({ type: "success", message: res.progress?.message || "Completed successfully." });
                    setProgressPct(100);
                    setTaskId(null);
                    clearInterval(interval);
                } else {
                    setStatus({ type: "processing", message: res.progress?.message || "Processing..." });
                    setProgressPct(res.progress?.percentage || 0);
                }
            } catch (err) {
                console.error("Failed to fetch progress:", err);
            }
        }, 1000);
        
        return () => clearInterval(interval);
    }, [taskId]);

    const handleFetch = useCallback(async () => {
        if (!form.year || !form.semester || !form.category || !form.option || form.data.length === 0) {
            setStatus({ type: "error", message: "Please complete all selections before fetching." });
            return;
        }

        setStatus({ type: "processing", message: "Starting fetch in background..." });
        setProgressPct(0);

        const yearLabel = years.find(y => y.value === form.year)?.label || form.year;
        const semLabel = semesters.find(s => s.value === form.semester)?.label || `Sem ${form.semester}`;
        const catLabel = categories.find(c => c.value === form.category)?.label || form.category;
        const optLabel = optionTypes.find(o => o.value === form.option)?.label || form.option;
        const dataLabels = form.data.map(d => dataOptions.find(o => o.value === d)?.label || d).join(", ");
        const autoLabel = form.label || `${catLabel} — ${dataLabels} (${yearLabel}, ${semLabel})`;

        try {
            const res = await fetchTimetable({
                year: form.year,
                semester: form.semester,
                category: form.category,
                option: form.option,
                data: form.data,
                label: autoLabel,
                strategy: form.strategy,
            });

            if (res.task_id) {
                setTaskId(res.task_id);
            }

            const entry = { label: autoLabel, time: new Date().toLocaleString(), status: "ingesting" };
            const newHistory = [entry, ...fetchHistory].slice(0, 20);
            setFetchHistory(newHistory);
            localStorage.setItem("tt_history", JSON.stringify(newHistory));
            
        } catch (err) {
            setStatus({ type: "error", message: err.message });
            setTaskId(null);
        }
    }, [form, years, semesters, categories, dataOptions, fetchHistory]);

    const toggleDataItem = (value) => {
        setForm(f => ({
            ...f,
            data: f.data.includes(value) ? f.data.filter(v => v !== value) : [...f.data, value],
        }));
    };

    const isReady = form.year && form.semester && form.category && form.data.length > 0;

    return (
        <div className="timetable-fetcher">
            <style>{`
                .timetable-fetcher { display: flex; flex-direction: column; gap: 28px; }
                .tt-header { display: flex; align-items: flex-start; gap: 16px; }
                .tt-header-icon { width: 48px; height: 48px; background: linear-gradient(135deg, var(--app-accent-strong), var(--app-accent)); border-radius: 12px; display: flex; align-items: center; justify-content: center; color: #fff; flex-shrink: 0; }
                .tt-header-text h2 { font-size: 20px; font-weight: 700; color: var(--app-text); margin: 0 0 4px; }
                .tt-header-text p { font-size: 13px; color: var(--app-muted); margin: 0; }

                .tt-card { background: var(--app-surface); border: 1px solid var(--app-border); border-radius: 14px; padding: 24px; box-shadow: var(--app-shadow); }
                .tt-card-title { font-size: 13px; font-weight: 600; color: var(--app-muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 20px; }

                .tt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
                @media(max-width: 640px) { .tt-grid { grid-template-columns: 1fr; } }

                .tt-field { display: flex; flex-direction: column; gap: 6px; }
                .tt-field label { font-size: 12px; font-weight: 600; color: var(--app-muted); letter-spacing: .04em; }
                .tt-select { width: 100%; background: var(--app-bg); border: 1px solid var(--app-border); color: var(--app-text); padding: 10px 14px; border-radius: 9px; font-size: 14px; appearance: none; cursor: pointer; outline: none; transition: border-color .2s; }
                .tt-select:focus { border-color: var(--app-accent); }
                .tt-select:disabled { opacity: .4; cursor: not-allowed; }

                .tt-loading-hint { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--app-muted); margin-top: 4px; }
                .spin-icon { animation: spin .8s linear infinite; }
                @keyframes spin { to { transform: rotate(360deg); } }

                .tt-data-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; max-height: 260px; overflow-y: auto; padding-right: 4px; }
                .tt-data-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border: 1px solid var(--app-border); border-radius: 8px; cursor: pointer; transition: all .15s; background: var(--app-bg); }
                .tt-data-item:hover { border-color: var(--app-accent); background: var(--app-panel); }
                .tt-data-item.selected { border-color: var(--app-accent); background: var(--app-accent-soft); }
                .tt-data-check { width: 18px; height: 18px; border: 1px solid var(--app-border-strong); border-radius: 4px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: all .15s; }
                .tt-data-item.selected .tt-data-check { background: var(--app-accent); border-color: var(--app-accent); color: #fff; }
                .tt-data-label { font-size: 13px; color: var(--app-text); line-height: 1.3; }

                .tt-input { width: 100%; background: var(--app-bg); border: 1px solid var(--app-border); color: var(--app-text); padding: 10px 14px; border-radius: 9px; font-size: 14px; outline: none; box-sizing: border-box; transition: border-color .2s; }
                .tt-input:focus { border-color: var(--app-accent); }
                .tt-input::placeholder { color: var(--app-faint); }

                .tt-btn { display: flex; align-items: center; gap: 10px; justify-content: center; padding: 13px 28px; border-radius: 10px; font-size: 15px; font-weight: 600; cursor: pointer; border: none; transition: all .2s; background: var(--app-text); color: var(--app-bg); }
                .tt-btn:hover:not(:disabled) { transform: translateY(-1px); opacity: 0.9; }
                .tt-btn:disabled { opacity: .5; cursor: not-allowed; transform: none; }

                .tt-status { display: flex; align-items: flex-start; gap: 10px; padding: 14px 16px; border-radius: 10px; font-size: 13px; line-height: 1.5; }
                .tt-status.success { background: var(--app-panel); border: 1px solid var(--app-border); color: var(--app-text); }
                .tt-status.error { background: var(--app-danger-soft); border: 1px solid var(--app-danger); color: var(--app-danger); }
                .tt-status.processing { background: var(--app-accent-soft); border: 1px solid var(--app-accent); color: var(--app-accent); }

                .tt-history { display: flex; flex-direction: column; gap: 8px; }
                .tt-history-item { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; background: var(--app-bg); border: 1px solid var(--app-border); border-radius: 9px; }
                .tt-history-label { font-size: 13px; color: var(--app-text); font-weight: 500; }
                .tt-history-time { font-size: 11px; color: var(--app-faint); }
                .tt-empty { font-size: 13px; color: var(--app-faint); text-align: center; padding: 24px; }
            `}</style>

            {/* Header */}
            <div className="tt-header">
                <div className="tt-header-icon"><CalendarIcon /></div>
                <div className="tt-header-text">
                    <h2>UDOM Timetable Fetcher</h2>
                    <p>Automatically fetch and sync timetables from ratiba.udom.ac.tz.</p>
                </div>
            </div>

            {/* Step 1 — Basic Selections */}
            <div className="tt-card">
                <p className="tt-card-title">Step 1 — Select Timetable</p>
                <div className="tt-grid">
                    {/* Academic Year */}
                    <div className="tt-field">
                        <label>Academic Year</label>
                        {loading.years ? (
                            <div className="tt-loading-hint"><SpinnerIcon /> Loading years…</div>
                        ) : (
                            <select className="tt-select" value={form.year} onChange={e => setForm(f => ({ ...f, year: e.target.value }))}>
                                <option value="">— Select Year —</option>
                                {years.map(y => <option key={y.value} value={y.value}>{y.label}</option>)}
                            </select>
                        )}
                    </div>

                    {/* Semester */}
                    <div className="tt-field">
                        <label>Semester</label>
                        <select className="tt-select" value={form.semester} onChange={e => setForm(f => ({ ...f, semester: e.target.value }))} disabled={!form.year || loading.semesters}>
                            <option value="">— Select Semester —</option>
                            {semesters.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                        </select>
                        {loading.semesters && <div className="tt-loading-hint"><SpinnerIcon /> Loading semesters…</div>}
                    </div>

                    {/* Timetable Category */}
                    <div className="tt-field">
                        <label>Timetable Category</label>
                        <select className="tt-select" value={form.category} onChange={e => setForm(f => ({ ...f, category: e.target.value }))} disabled={!form.semester || loading.categories}>
                            <option value="">— Select Category —</option>
                            {categories.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
                        </select>
                        {loading.categories && <div className="tt-loading-hint"><SpinnerIcon /> Loading categories…</div>}
                    </div>

                    {/* Option Type — live from UDOM */}
                    <div className="tt-field">
                        <label>Download Option</label>
                        <select
                            className="tt-select"
                            value={form.option}
                            onChange={e => setForm(f => ({ ...f, option: e.target.value, data: [] }))}
                            disabled={!form.category || loading.optionTypes}
                        >
                            <option value="">— Select Option —</option>
                            {optionTypes.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                        {loading.optionTypes && <div className="tt-loading-hint"><SpinnerIcon /> Loading options…</div>}
                    </div>
                </div>
            </div>

            {/* Step 2 — Data Selection */}
            {form.option && (
                <div className="tt-card">
                    <p className="tt-card-title">
                        Step 2 — Select {optionTypes.find(o => o.value === form.option)?.label || form.option}
                        {form.data.length > 0 && <span style={{ marginLeft: 8, color: "#3b82f6" }}>{form.data.length} selected</span>}
                    </p>
                    {loading.dataOptions ? (
                        <div className="tt-loading-hint"><SpinnerIcon /> Loading options…</div>
                    ) : dataOptions.length === 0 ? (
                        <p className="tt-empty">No options found. Try a different selection above.</p>
                    ) : (
                        <>
                            <div style={{ display: "flex", gap: "10px", marginBottom: "16px", alignItems: "center" }}>
                                <input 
                                    type="text" 
                                    placeholder="Filter programmes (e.g. Computer Science)..." 
                                    style={{ flex: 1, padding: "8px 12px", background: "#131e2d", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "6px", color: "white" }}
                                    onChange={(e) => {
                                        const term = e.target.value.toLowerCase();
                                        document.querySelectorAll('.tt-data-item').forEach(el => {
                                            const label = el.querySelector('.tt-data-label').innerText.toLowerCase();
                                            el.style.display = label.includes(term) ? "flex" : "none";
                                        });
                                    }}
                                />
                                <button 
                                    className="tt-btn-secondary" 
                                    style={{ padding: "8px 16px" }}
                                    onClick={() => {
                                        const visibleOptions = Array.from(document.querySelectorAll('.tt-data-item'))
                                            .filter(el => el.style.display !== 'none')
                                            .map(el => el.getAttribute('data-value'));
                                        setForm(f => ({ ...f, data: [...new Set([...f.data, ...visibleOptions])] }));
                                    }}
                                >
                                    Select All Visible
                                </button>
                                <button 
                                    className="tt-btn-secondary" 
                                    style={{ padding: "8px 16px" }}
                                    onClick={() => setForm(f => ({ ...f, data: [] }))}
                                >
                                    Clear Selection
                                </button>
                            </div>
                            <div className="tt-data-grid">
                                {dataOptions.map(opt => (
                                <div
                                    key={opt.value}
                                    data-value={opt.value}
                                    className={`tt-data-item${form.data.includes(opt.value) ? " selected" : ""}`}
                                    onClick={() => toggleDataItem(opt.value)}
                                    role="checkbox"
                                    aria-checked={form.data.includes(opt.value)}
                                    tabIndex={0}
                                    onKeyDown={e => e.key === " " && toggleDataItem(opt.value)}
                                >
                                    <div className="tt-data-check">
                                        {form.data.includes(opt.value) && <CheckIcon />}
                                    </div>
                                    <span className="tt-data-label">{opt.label}</span>
                                </div>
                            ))}
                        </div>
                        </>
                    )}
                </div>
            )}

            {/* Step 3 — Strategy + Label + Fetch */}
            {isReady && (
                <div className="tt-card">
                    <p className="tt-card-title">Step 3 — Fetch &amp; Ingest</p>
                    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                        {/* Extraction Strategy */}
                        <div className="tt-field">
                            <label>Text Extraction Strategy</label>
                            <select
                                id="select-extraction-strategy"
                                className="tt-select"
                                value={form.strategy}
                                onChange={e => setForm(f => ({ ...f, strategy: e.target.value }))}
                            >
                                {STRATEGIES.map(s => (
                                    <option key={s.value} value={s.value}>{s.label}</option>
                                ))}
                            </select>
                            <small style={{ color: "#64748b", fontSize: "12px", marginTop: 4 }}>
                                {STRATEGIES.find(s => s.value === form.strategy)?.desc}
                            </small>
                        </div>

                        <div className="tt-field">
                            <label>Document Label (optional)</label>
                            <input
                                className="tt-input"
                                type="text"
                                placeholder="e.g. BSc Computer Science Year 3 — Semester 1 Teaching Timetable"
                                value={form.label}
                                onChange={e => setForm(f => ({ ...f, label: e.target.value }))}
                            />
                        </div>
                        <button id="btn-fetch-timetable" className="tt-btn" onClick={handleFetch} disabled={status.type === "processing"}>
                            {status.type === "processing" ? <SpinnerIcon /> : <DownloadIcon />}
                            {status.type === "processing" ? "Fetching…" : "Fetch & Ingest Timetable"}
                        </button>
                    </div>
                </div>
            )}

            {/* Status */}
            {status.message && (
                <div className={`tt-status ${status.type}`} style={{ flexDirection: "column" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                        {status.type === "processing" ? <SpinnerIcon /> : <AlertIcon />}
                        <span>{status.message}</span>
                    </div>
                    {status.type === "processing" && (
                        <div style={{ width: "100%", height: "4px", background: "var(--app-border)", borderRadius: "2px", marginTop: "10px", overflow: "hidden" }}>
                            <div style={{ width: `${progressPct}%`, height: "100%", background: "var(--app-accent)", transition: "width 0.3s ease" }} />
                        </div>
                    )}
                </div>
            )}

            {/* History */}
            <div className="tt-card">
                <p className="tt-card-title">Fetch History (this session)</p>
                {fetchHistory.length === 0 ? (
                    <p className="tt-empty">No timetables fetched yet in this browser session.</p>
                ) : (
                    <div className="tt-history">
                        {fetchHistory.map((item, idx) => (
                            <div className="tt-history-item" key={idx}>
                                <div>
                                    <div className="tt-history-label">{item.label}</div>
                                    <div className="tt-history-time">{item.time}</div>
                                </div>
                                <span style={{ fontSize: "11px", color: "#34d399", background: "rgba(52,211,153,0.1)", padding: "3px 8px", borderRadius: "6px" }}>
                                    ingested
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
