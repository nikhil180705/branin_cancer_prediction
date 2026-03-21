export default function Sidebar({ page, navigate, hasResults, mobileOpen, setMobileOpen }) {
  const items = [
    { id: 'dashboard', label: 'Dashboard', icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg> },
    { id: 'upload', label: 'Upload MRI', icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> },
    ...(hasResults ? [{ id: 'results', label: 'View Results', icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/></svg> }] : []),
    { id: 'compare', label: 'Compare Scans', icon: <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> }
  ]

  return (
    <>
      <aside className={`sidebar${mobileOpen ? ' open' : ''}`}>
        <div className="sidebar-head">
          <svg className="brand-icon" viewBox="0 0 34 34" fill="none">
            <circle cx="17" cy="17" r="15" stroke="#3b82f6" strokeWidth="2.5"/>
            <path d="M11 17C11 13.1 14.1 10 18 10C21.9 10 24 13 24 17" stroke="#60a5fa" strokeWidth="2" strokeLinecap="round"/>
            <circle cx="17" cy="17" r="3.5" fill="#3b82f6"/>
          </svg>
          <div>
            <h1>Brain Tumor</h1>
            <span>AI Analyzer</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {items.map(it => (
            <button key={it.id} className={page === it.id ? 'active' : ''} onClick={() => navigate(it.id)}>
              {it.icon}{it.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">⚕️ AI-assisted tool — Not a medical diagnosis</div>
      </aside>

      <div className={`sidebar-overlay${mobileOpen ? ' open' : ''}`}
           onClick={() => setMobileOpen(false)} />
    </>
  )
}
