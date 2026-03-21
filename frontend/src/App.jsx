import { useState, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Results from './pages/Results'
import Compare from './pages/Compare'

export default function App() {
  const [page, setPage] = useState('dashboard')
  const [result, setResult] = useState(null)
  const [recentCases, setRecentCases] = useState(() => {
    try { return JSON.parse(localStorage.getItem('recentCases') || '[]') } catch { return [] }
  })
  const [mobileOpen, setMobileOpen] = useState(false)

  const navigate = useCallback((p) => {
    setPage(p)
    setMobileOpen(false)
  }, [])

  const addCase = useCallback((data) => {
    const c = {
      id: Date.now(),
      filename: data.filename || 'Unknown',
      type: data.prediction.display_name,
      hasTumor: data.prediction.has_tumor,
      confidence: data.prediction.confidence,
      risk: data.risk_level.risk_level,
      time: new Date().toLocaleTimeString()
    }
    setRecentCases(prev => {
      const next = [c, ...prev].slice(0, 10)
      localStorage.setItem('recentCases', JSON.stringify(next))
      return next
    })
  }, [])

  const handleAnalysisComplete = useCallback((data) => {
    setResult(data)
    addCase(data)
    setPage('results')
  }, [addCase])

  return (
    <div className="app-layout">
      <Sidebar page={page} navigate={navigate} hasResults={!!result}
               mobileOpen={mobileOpen} setMobileOpen={setMobileOpen} />
      
      {/* Mobile bar */}
      <div className="mobile-bar">
        <button onClick={() => setMobileOpen(true)}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>
        <span>Brain Tumor AI Analyzer</span>
      </div>

      <main className="main-content">
        {page === 'dashboard' && <Dashboard navigate={navigate} cases={recentCases} />}
        {page === 'upload' && <Upload onComplete={handleAnalysisComplete} />}
        {page === 'results' && <Results data={result} onNewScan={() => { setPage('upload') }} />}
        {page === 'compare' && <Compare />}
      </main>
    </div>
  )
}
