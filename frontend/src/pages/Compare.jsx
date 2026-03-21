import { useState, useRef, useCallback } from 'react'
import './Compare.css'

export default function Compare() {
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef()

  const handleFiles = useCallback((fl) => {
    const arr = Array.from(fl)
    if (arr.length < 2) { setError('Select at least 2 files'); return }
    if (arr.length > 5) { setError('Maximum 5 files allowed'); return }
    setError(null)
    setFiles(arr)
  }, [])

  const runCompare = async () => {
    setLoading(true); setError(null)
    const fd = new FormData()
    files.forEach(f => fd.append('files', f))

    try {
      const res = await fetch('/api/compare', { method: 'POST', body: fd })
      if (!res.ok) { const e = await res.json(); throw new Error(e.error) }
      setResults(await res.json())
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  const reset = () => { setFiles([]); setResults(null); setError(null); if (inputRef.current) inputRef.current.value = '' }

  if (loading) {
    return (
      <div className="fade-in">
        <div className="page-head"><h2>Compare Scans</h2></div>
        <div className="spinner-wrap slide-up">
          <div className="spinner-ring" />
          <h3>Comparing scans...</h3>
          <p>Analyzing each scan and computing progression</p>
          <div className="progress-dots"><span/><span/><span/></div>
        </div>
      </div>
    )
  }

  if (results) {
    const prog = results.comparison.progression?.toLowerCase()
    const progClass = prog === 'growth' ? 'red' : prog === 'stable' ? 'green' : prog === 'reduced' ? 'teal' : 'amber'

    return (
      <div className="fade-in">
        <div className="page-head">
          <h2>Comparison Results</h2>
          <p>{results.scans.length} scans analyzed</p>
        </div>

        {/* Progression Card */}
        <div className="prog-card slide-up">
          <span className="prog-label">Tumor Progression</span>
          <span className={`prog-val color-${progClass}`}>{results.comparison.progression}</span>
          <p className="prog-summary">{results.comparison.summary}</p>
        </div>

        {/* Scan Cards Grid */}
        <div className="compare-grid">
          {results.scans.map((s, i) => {
            const rskCls = s.risk_level === 'High' ? 'badge-red' : s.risk_level === 'Medium' ? 'badge-amber' : 'badge-green'
            return (
              <div key={i} className="compare-card slide-up" style={{ animationDelay: `${i * .08}s` }}>
                <div className="cc-head">
                  <h4>{s.scan_label}</h4>
                  <span className={`badge ${rskCls}`}>{s.risk_level}</span>
                </div>
                <div className="cc-body">
                  <img className="cc-img" src={`data:image/png;base64,${s.images.overlay}`} alt={s.scan_label} />
                  <div className="cc-row"><span className="lbl">File</span><span className="val">{s.original_filename}</span></div>
                  <div className="cc-row"><span className="lbl">Type</span><span className="val">{s.display_name}</span></div>
                  <div className="cc-row"><span className="lbl">Confidence</span><span className="val">{s.confidence}%</span></div>
                  <div className="cc-row"><span className="lbl">Size</span><span className="val">{s.size_category}</span></div>
                  <div className="cc-row"><span className="lbl">Activation</span><span className="val">{s.activation_percentage}%</span></div>
                </div>
              </div>
            )
          })}
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1.2rem' }}>
          <button className="btn btn-outline" onClick={reset}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            New Comparison
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="fade-in">
      <div className="page-head">
        <h2>Compare Scans</h2>
        <p>Upload 2–5 MRI scans to track tumor progression over time</p>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="upload-zone" onClick={() => inputRef.current?.click()}>
        <div className="upload-icon-wrap">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><path d="M7 14v3a1 1 0 001 1h3"/><path d="M17 14v3a1 1 0 01-1 1h-3"/></svg>
        </div>
        <h3>Select MRI Scans for Comparison</h3>
        <p>Choose 2–5 images</p>
      </div>

      <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif" multiple hidden
             onChange={e => handleFiles(e.target.files)} />

      {files.length > 0 && (
        <div className="selected-strip slide-up">
          <div className="chip-wrap">
            {files.map((f, i) => <span key={i} className="file-chip">📄 {f.name}</span>)}
          </div>
          <button className="btn btn-primary" onClick={runCompare}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            Compare Scans
          </button>
        </div>
      )}
    </div>
  )
}
