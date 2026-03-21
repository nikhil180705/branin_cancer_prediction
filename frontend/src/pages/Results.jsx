import { useState, useEffect, useRef } from 'react'
import './Results.css'

function ConfidenceRing({ value }) {
  const r = 34, c = 2 * Math.PI * r
  const offset = c * (1 - value / 100)
  const ref = useRef()

  useEffect(() => {
    const el = ref.current
    if (el) { el.style.transition = 'none'; el.setAttribute('stroke-dashoffset', c)
      requestAnimationFrame(() => { el.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)'; el.setAttribute('stroke-dashoffset', offset) }) }
  }, [value, offset, c])

  return (
    <div className="conf-ring-wrap">
      <svg viewBox="0 0 80 80" width="68" height="68">
        <circle cx="40" cy="40" r={r} stroke="#e2e8f0" strokeWidth="6" fill="none" />
        <circle ref={ref} cx="40" cy="40" r={r} stroke="#3b82f6" strokeWidth="6" fill="none"
                strokeDasharray={c} strokeDashoffset={c} strokeLinecap="round" transform="rotate(-90 40 40)" />
      </svg>
      <span className="conf-val">{value}%</span>
    </div>
  )
}

function ProbBar({ name, value, delay }) {
  const [w, setW] = useState(0)
  useEffect(() => { const t = setTimeout(() => setW(value), delay); return () => clearTimeout(t) }, [value, delay])
  return (
    <div className="prob-row">
      <span className="prob-name">{name}</span>
      <div className="prob-track"><div className="prob-fill" style={{ width: w + '%' }} /></div>
      <span className="prob-val">{value}%</span>
    </div>
  )
}

export default function Results({ data, onNewScan }) {
  const [report, setReport] = useState(data?.report?.report || '')
  const [reportMethod, setReportMethod] = useState(data?.report?.method || 'template')
  const [llmLoading, setLlmLoading] = useState(false)

  if (!data) return <div className="fade-in"><p>No results yet. Upload an MRI to analyze.</p></div>

  const { prediction: pred, tumor_size, risk_level, images } = data

  const badgeClass = (val) => {
    const v = val?.toLowerCase()
    if (['high', 'large'].includes(v)) return 'badge-red'
    if (['medium'].includes(v)) return 'badge-amber'
    return 'badge-green'
  }

  const generateLLM = async () => {
    setLlmLoading(true)
    try {
      const res = await fetch('/api/generate-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tumor_detected: pred.has_tumor,
          tumor_type: pred.class,
          confidence: pred.confidence / 100,
          tumor_size: tumor_size.size_category,
          risk_level: risk_level.risk_level
        })
      })
      const d = await res.json()
      setReport(d.report)
      setReportMethod(d.method)
    } catch { /* ignore */ }
    setLlmLoading(false)
  }

  return (
    <div className="fade-in">
      <div className="page-head">
        <h2>Analysis Results</h2>
        <p>{data.filename}</p>
      </div>

      {/* Summary Cards */}
      <div className="grid-5 summary-row">
        <div className={`summary-card ${pred.has_tumor ? 'border-red' : 'border-green'}`}>
          <span className="s-label">Detection</span>
          <span className="s-value">{pred.has_tumor ? '⚠️ Tumor' : '✅ Clear'}</span>
        </div>
        <div className="summary-card">
          <span className="s-label">Tumor Type</span>
          <span className="s-value">{pred.display_name}</span>
        </div>
        <div className="summary-card">
          <span className="s-label">Confidence</span>
          <ConfidenceRing value={pred.confidence} />
        </div>
        <div className="summary-card">
          <span className="s-label">Tumor Size</span>
          <span className={`badge ${badgeClass(tumor_size.size_category)}`}>{tumor_size.size_category}</span>
        </div>
        <div className="summary-card">
          <span className="s-label">Risk Level</span>
          <span className={`badge ${badgeClass(risk_level.risk_level)}`}>{risk_level.risk_level}</span>
        </div>
      </div>

      {/* MRI Visualization */}
      <div className="card slide-up">
        <div className="card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
          <h3>MRI Visualization</h3>
        </div>
        <div className="grid-3 mri-grid">
          {[['Original MRI', images.original], ['Grad-CAM Heatmap', images.heatmap], ['Overlay', images.overlay]].map(([title, b64]) => (
            <div key={title} className="mri-card">
              <h4>{title}</h4>
              <div className="mri-frame"><img src={`data:image/png;base64,${b64}`} alt={title} /></div>
            </div>
          ))}
        </div>
      </div>

      {/* Probabilities */}
      <div className="card slide-up">
        <div className="card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" strokeWidth="2"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
          <h3>Class Probabilities</h3>
        </div>
        {Object.entries(pred.probabilities).map(([name, val], i) => (
          <ProbBar key={name} name={name} value={val} delay={i * 120 + 200} />
        ))}
      </div>

      {/* AI Report */}
      <div className="card slide-up report-card">
        <div className="card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" strokeWidth="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z"/><path d="M14 2v6h6"/></svg>
          <h3>AI Medical Report</h3>
          <button className="btn btn-sm btn-outline" onClick={generateLLM} disabled={llmLoading}>
            {llmLoading ? '⏳ Generating...' : '✨ Generate LLM Report'}
          </button>
        </div>
        <div className="report-box">
          <pre className="report-text">{report}</pre>
        </div>
        <div className="report-meta">
          {reportMethod === 'llm' ? '✨ Generated by Qwen2.5-0.5B-Instruct' : '📋 Template-based report'}
        </div>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1.2rem' }}>
        <button className="btn btn-outline" onClick={onNewScan}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          Analyze New Scan
        </button>
      </div>
    </div>
  )
}
