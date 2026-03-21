import './Dashboard.css'

const KNOWLEDGE = [
  { type: 'Glioma', color: 'glioma', desc: 'Originates from glial cells that support neurons. Can range from low-grade (slow-growing) to high-grade (aggressive, such as glioblastoma).', symptoms: 'Headaches, seizures, memory loss', risk: 'Can be fast-growing and aggressive', treatment: 'Surgery, radiation, chemotherapy' },
  { type: 'Meningioma', color: 'meningioma', desc: 'Arises from the meninges — protective membranes surrounding the brain. Usually slow-growing and benign.', symptoms: 'Vision problems, headaches, weakness', risk: 'Mostly benign, can compress brain', treatment: 'Monitoring, surgery if symptomatic' },
  { type: 'Pituitary', color: 'pituitary', desc: 'Develops in the pituitary gland at the base of the brain. Can affect hormone production and vision.', symptoms: 'Vision changes, hormonal imbalances', risk: 'Usually benign and treatable', treatment: 'Medication, surgery, radiation' }
]

const STEPS = [
  { n: 1, title: 'Upload', text: 'Upload a brain MRI image (JPG, PNG, BMP, or TIFF)' },
  { n: 2, title: 'Analyze', text: 'AI classifies the tumor and generates a Grad-CAM heatmap' },
  { n: 3, title: 'Review', text: 'View tumor type, size, risk, confidence, and AI report' },
  { n: 4, title: 'Track', text: 'Compare scans over time to monitor progression' }
]

export default function Dashboard({ navigate, cases }) {
  return (
    <div className="fade-in">
      <div className="page-head">
        <h2>Dashboard</h2>
        <p>Welcome to the Brain Tumor AI Analysis System</p>
      </div>

      {/* Quick Actions */}
      <div className="grid-2" style={{ marginBottom: '1.2rem' }}>
        <button className="action-card" onClick={() => navigate('upload')}>
          <div className="action-icon blue">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          </div>
          <h3>Upload MRI Scan</h3>
          <p>Analyze a new brain MRI image</p>
        </button>
        <button className="action-card" onClick={() => navigate('compare')}>
          <div className="action-icon teal">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          </div>
          <h3>Compare Scans</h3>
          <p>Track tumor progression over time</p>
        </button>
      </div>

      {/* Instructions */}
      <div className="card">
        <div className="card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
          <h3>How to Use</h3>
        </div>
        <div className="grid-4">
          {STEPS.map(s => (
            <div key={s.n} className="step-card">
              <span className="step-num">{s.n}</span>
              <h4>{s.title}</h4>
              <p>{s.text}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Knowledge */}
      <div className="card">
        <div className="card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" strokeWidth="2"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>
          <h3>Brain Tumor Knowledge</h3>
        </div>
        <div className="grid-3">
          {KNOWLEDGE.map(k => (
            <div key={k.type} className="knowledge-card">
              <div className={`knowledge-hdr ${k.color}`}>{k.type}</div>
              <div className="knowledge-body">
                <p>{k.desc}</p>
                <ul>
                  <li><strong>Symptoms:</strong> {k.symptoms}</li>
                  <li><strong>Risk:</strong> {k.risk}</li>
                  <li><strong>Treatment:</strong> {k.treatment}</li>
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Cases */}
      <div className="card">
        <div className="card-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--blue)" strokeWidth="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
          <h3>Recent Cases</h3>
        </div>
        {cases.length === 0 ? (
          <p style={{ textAlign: 'center', padding: '1.5rem', color: 'var(--text-muted)', fontSize: '.85rem' }}>
            No scans analyzed yet. Upload an MRI to get started.
          </p>
        ) : (
          <div className="case-list">
            {cases.map(c => (
              <div key={c.id} className="case-row">
                <div className={`case-dot ${c.hasTumor ? 'red' : 'green'}`} />
                <div className="case-info">
                  <div className="case-name">{c.filename}</div>
                  <div className="case-detail">{c.type} · {c.confidence}% · {c.time}</div>
                </div>
                <span className={`badge ${c.risk === 'High' ? 'badge-red' : c.risk === 'Medium' ? 'badge-amber' : 'badge-green'}`}>
                  {c.risk}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
