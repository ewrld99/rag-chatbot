import { useState, useEffect, useCallback } from "react";
import {
    getYears,
    getSemesters,
    getCategories,
    getOptionTypes,
    getDataOptions,
    fetchTimetable,
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

    const handleFetch = useCallback(async () => {
        if (!form.year || !form.semester || !form.category || !form.option || form.data.length === 0) {
            setStatus({ type: "error", message: "Please complete all selections before fetching." });
            return;
        }

        setStatus({ type: "processing", message: "Starting fetch in background..." });

        const yearLabel = years.find(y => y.value === form.year)?.label || form.year;
        const semLabel = semesters.find(s => s.value === form.semester)?.label || `Sem ${form.semester}`;
        const catLabel = categories.find(c => c.value === form.category)?.label || form.category;
        const optLabel = optionTypes.find(o => o.value === form.option)?.label || form.option;
        const dataLabels = form.data.map(d => dataOptions.find(o => o.value === d)?.label || d).join(", ");
        const autoLabel = form.label || `${catLabel} — ${dataLabels} (${yearLabel}, ${semLabel})`;

        try {
            await fetchTimetable({
                year: form.year,
                semester: form.semester,
                category: form.category,
                option: form.option,
                data: form.data,
                label: autoLabel,
                strategy: form.strategy,
            });

            const entry = { label: autoLabel, time: new Date().toLocaleString(), status: "ingesting" };
            const newHistory = [entry, ...fetchHistory].slice(0, 20);
            setFetchHistory(newHistory);
            localStorage.setItem("tt_history", JSON.stringify(newHistory));

            setStatus({
                type: "success",
                message: `✓ "${autoLabel}" is being downloaded and ingested. It will appear in Documents once complete.`,
            });
        } catch (err) {
            setStatus({ type: "error", message: err.message });
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
                .tt-header-icon { width: 48px; height: 48px; background: linear-gradient(135deg, #1e40af, #3b82f6); border-radius: 12px; display: flex; align-items: center; justify-content: center; color: #fff; flex-shrink: 0; }
                .tt-header-text h2 { font-size: 20px; font-weight: 700; color: #f1f5f9; margin: 0 0 4px; }
                .tt-header-text p { font-size: 13px; color: #64748b; margin: 0; }

                .tt-card { background: #0f1923; border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 24px; }
                .tt-card-title { font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 20px; }

                .tt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
                @media(max-width: 640px) { .tt-grid { grid-template-columns: 1fr; } }

                .tt-field { display: flex; flex-direction: column; gap: 6px; }
                .tt-field label { font-size: 12px; font-weight: 600; color: #94a3b8; letter-spacing: .04em; }
                .tt-select { width: 100%; background: #131e2d; border: 1px solid rgba(255,255,255,0.09); color: #e2e8f0; padding: 10px 14px; border-radius: 9px; font-size: 14px; appearance: none; cursor: pointer; outline: none; transition: border-color .2s; }
                .tt-select:focus { border-color: #3b82f6; }
                .tt-select:disabled { opacity: .4; cursor: not-allowed; }

                .tt-loading-hint { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #64748b; margin-top: 4px; }
                .spin-icon { animation: spin .8s linear infinite; }
                @keyframes spin { to { transform: rotate(360deg); } }

                .tt-data-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; max-height: 260px; overflow-y: auto; padding-right: 4px; }
                .tt-data-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; cursor: pointer; transition: all .15s; background: #131e2d; }
                .tt-data-item:hover { border-color: #3b82f6; background: rgba(59,130,246,0.06); }
                .tt-data-item.selected { border-color: #3b82f6; background: rgba(59,130,246,0.12); }
                .tt-data-check { width: 18px; height: 18px; border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: all .15s; }
                .tt-data-item.selected .tt-data-check { background: #3b82f6; border-color: #3b82f6; color: #fff; }
                .tt-data-label { font-size: 13px; color: #cbd5e1; line-height: 1.3; }

                .tt-input { width: 100%; background: #131e2d; border: 1px solid rgba(255,255,255,0.09); color: #e2e8f0; padding: 10px 14px; border-radius: 9px; font-size: 14px; outline: none; box-sizing: border-box; transition: border-color .2s; }
                .tt-input:focus { border-color: #3b82f6; }
                .tt-input::placeholder { color: #4b5563; }

                .tt-btn { display: flex; align-items: center; gap: 10px; justify-content: center; padding: 13px 28px; border-radius: 10px; font-size: 15px; font-weight: 600; cursor: pointer; border: none; transition: all .2s; background: linear-gradient(135deg, #1e40af, #3b82f6); color: #fff; box-shadow: 0 4px 16px rgba(59,130,246,0.25); }
                .tt-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 24px rgba(59,130,246,0.35); }
                .tt-btn:disabled { opacity: .5; cursor: not-allowed; transform: none; }

                .tt-status { display: flex; align-items: flex-start; gap: 10px; padding: 14px 16px; border-radius: 10px; font-size: 13px; line-height: 1.5; }
                .tt-status.success { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.2); color: #34d399; }
                .tt-status.error { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); color: #f87171; }
                .tt-status.processing { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); color: #60a5fa; }

                .tt-history { display: flex; flex-direction: column; gap: 8px; }
                .tt-history-item { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; background: #131e2d; border: 1px solid rgba(255,255,255,0.06); border-radius: 9px; }
                .tt-history-label { font-size: 13px; color: #cbd5e1; font-weight: 500; }
                .tt-history-time { font-size: 11px; color: #475569; }
                .tt-empty { font-size: 13px; color: #475569; text-align: center; padding: 24px; }
            `}</style>

            {/* Header */}
            <div className="tt-header">
                <div className="tt-header-icon"><CalendarIcon /></div>
                <div className="tt-header-text">
                    <h2>UDOM Timetable Fetcher</h2>
                    <p>Select a timetable from ratiba.udom.ac.tz and ingest it into the RAG knowledge base automatically.</p>
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
                        <div className="tt-data-grid">
                            {dataOptions.map(opt => (
                                <div
                                    key={opt.value}
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
                <div className={`tt-status ${status.type}`}>
                    {status.type === "processing" ? <SpinnerIcon /> : <AlertIcon />}
                    <span>{status.message}</span>
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
