import { useState, useRef, useCallback } from 'react'
import './Upload.css'

export default function Upload({ onComplete }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef()

  const handleFile = useCallback((f) => {
    if (!f) return
    const ext = f.name.split('.').pop().toLowerCase()
    if (!['jpg','jpeg','png','bmp','tiff','tif'].includes(ext)) {
      setError('Unsupported file type. Please upload JPG, PNG, BMP, or TIFF.'); return
    }
    setError(null)
    setFile(f)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const analyze = useCallback(async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    const fd = new FormData()
    fd.append('file', file)

    try {
      const res = await fetch('/api/predict', { method: 'POST', body: fd })
      if (!res.ok) { const e = await res.json(); throw new Error(e.error || 'Analysis failed') }
      const data = await res.json()
      onComplete(data)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }, [file, onComplete])

  const reset = () => { setFile(null); setPreview(null); setError(null); inputRef.current.value = '' }

  if (loading) {
    return (
      <div className="fade-in">
        <div className="page-head"><h2>Upload MRI Scan</h2></div>
        <div className="spinner-wrap slide-up">
          <div className="spinner-ring" />
          <h3>Analyzing MRI scan...</h3>
          <p>Running classification, generating heatmap, computing risk</p>
          <div className="progress-dots"><span/><span/><span/></div>
        </div>
      </div>
    )
  }

  return (
    <div className="fade-in">
      <div className="page-head">
        <h2>Upload MRI Scan</h2>
        <p>Upload a brain MRI image for AI-powered analysis</p>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {!file ? (
        <div className={`upload-zone${dragOver ? ' over' : ''}`}
             onClick={() => inputRef.current?.click()}
             onDragOver={e => { e.preventDefault(); setDragOver(true) }}
             onDragLeave={() => setDragOver(false)}
             onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}>
          <div className="upload-icon-wrap">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <h3>Drag & Drop MRI Image</h3>
          <p>or click to browse your files</p>
          <span className="file-types-tag">Supports: JPG, PNG, BMP, TIFF</span>
        </div>
      ) : (
        <div className="preview-card slide-up">
          <img src={preview} alt="MRI preview" className="preview-img" />
          <div className="preview-right">
            <div className="preview-name">{file.name}</div>
            <div className="preview-size">{(file.size / 1024).toFixed(0)} KB</div>
            <div className="preview-actions">
              <button className="btn btn-outline" onClick={reset}>Change File</button>
              <button className="btn btn-primary" onClick={analyze}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>
                Analyze Scan
              </button>
            </div>
          </div>
        </div>
      )}

      <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif" hidden
             onChange={e => handleFile(e.target.files[0])} />
    </div>
  )
}
