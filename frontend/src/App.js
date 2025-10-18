import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function Toast({ type = 'info', message, onClose }) {
  if (!message) return null;
  return (
    <div className={`toast toast-${type}`} role="status">
      <span>{message}</span>
      <button className="toast-close" onClick={onClose} aria-label="Close notification">✕</button>
    </div>
  );
}

function LoadingOverlay({ show, label = 'Dolgozunk...' }) {
  if (!show) return null;
  return (
    <div className="overlay">
      <div className="spinner" aria-label="Loading" />
      <div className="overlay-label">{label}</div>
    </div>
  );
}

function App() {
  const [documentFile, setDocumentFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [indexed, setIndexed] = useState(false);
  const [toast, setToast] = useState({ type: 'info', message: '' });
  const [uploadProgress, setUploadProgress] = useState(0);
  const dropRef = useRef(null);

  const canAsk = indexed && question.trim().length > 0 && !loading;

  useEffect(() => {
    const el = dropRef.current;
    if (!el) return;
    const preventDefaults = (e) => { e.preventDefault(); e.stopPropagation(); };
    const highlight = () => el.classList.add('dropzone-hover');
    const unhighlight = () => el.classList.remove('dropzone-hover');
    const handleDrop = (e) => {
      const files = e.dataTransfer.files;
      if (files && files[0]) handleFile(files[0]);
    };
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => el.addEventListener(ev, preventDefaults));
    ['dragenter', 'dragover'].forEach(ev => el.addEventListener(ev, highlight));
    ['dragleave', 'drop'].forEach(ev => el.addEventListener(ev, unhighlight));
    el.addEventListener('drop', handleDrop);
    return () => {
      ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => el.removeEventListener(ev, preventDefaults));
      ['dragenter', 'dragover'].forEach(ev => el.removeEventListener(ev, highlight));
      ['dragleave', 'drop'].forEach(ev => el.removeEventListener(ev, unhighlight));
      el.removeEventListener('drop', handleDrop);
    };
  }, []);

  const handleFile = (file) => {
    if (!file) return;
    const isTxt = file.name.toLowerCase().endsWith('.txt');
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (!isTxt) {
      setToast({ type: 'warning', message: 'Csak .txt fájlok támogatottak.' });
      return;
    }
    if (file.size > maxSize) {
      setToast({ type: 'warning', message: 'A fájl túl nagy. Maximum 5MB.' });
      return;
    }
    setDocumentFile(file);
    setToast({ type: 'info', message: `Kiválasztva: ${file.name}` });
  };

  const handleUpload = async () => {
    if (!documentFile) {
      setToast({ type: 'warning', message: 'Please choose a .txt file first.' });
      return;
    }
    setLoading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', documentFile);
    try {
      await axios.post(`${API_BASE}/upload_document/`, formData, {
        onUploadProgress: (evt) => {
          if (!evt.total) return;
          const percent = Math.round((evt.loaded * 100) / evt.total);
          setUploadProgress(percent);
        }
      });
      setIndexed(true);
      setToast({ type: 'success', message: 'Dokumentum feltöltve és indexelve.' });
    } catch (error) {
      const msg = error?.response?.data?.detail || error.message;
      setToast({ type: 'error', message: `Feltöltés sikertelen: ${msg}` });
    }
    setLoading(false);
  };

  const handleQuery = async () => {
    if (!canAsk) return;
    setLoading(true);
    setToast({ type: 'info', message: 'Válasz készítése...' });
    try {
      const res = await axios.post(`${API_BASE}/query/`, { question });
      setResponse(res.data);
      setToast({ type: 'success', message: 'Válasz elkészült.' });
    } catch (error) {
      const msg = error?.response?.data?.detail || error.message;
      setToast({ type: 'error', message: `Lekérdezés sikertelen: ${msg}` });
    }
    setLoading(false);
  };

  const copyAnswer = async () => {
    if (!response) return;
    const full = `Válasz: ${response.answer}`;
    await navigator.clipboard.writeText(full);
    setToast({ type: 'success', message: 'Vágólapra másolva.' });
  };

  const disabledUpload = loading || !documentFile;

  return (
    <div className="app-shell">
      <Toast type={toast.type} message={toast.message} onClose={() => setToast({ ...toast, message: '' })} />
      <LoadingOverlay show={loading} label={indexed ? 'Thinking...' : 'Uploading...'} />

      <header className="app-header">
        <div className="brand">
          <span className="logo" aria-hidden>🔎</span>
          <h1>Kvíz Dokumentum Kereső</h1>
        </div>
        <div className="status">
          {indexed ? <span className="badge badge-success">Indexelve</span> : <span className="badge">Dokumentumra vár</span>}
        </div>
      </header>

      <main className="content">
        <section className="card">
          <h2>Dokumentum feltöltése</h2>
          <div ref={dropRef} className="dropzone" role="button" tabIndex={0} onClick={() => document.getElementById('file-input').click()} onKeyDown={(e) => { if (e.key === 'Enter') document.getElementById('file-input').click(); }}>
            <input id="file-input" type="file" accept=".txt" onChange={(e) => handleFile(e.target.files[0])} hidden />
            <div className="dropzone-inner">
              <div className="drop-icon">📄</div>
              <div>
                <div className="drop-title">Húzd ide a .txt fájlod</div>
                <div className="drop-subtitle">vagy kattints a böngészéshez (max 5MB)</div>
              </div>
            </div>
          </div>
          {documentFile && (
            <div className="file-row">
              <div className="file-name" title={documentFile.name}>{documentFile.name}</div>
              <button className="btn btn-secondary" onClick={() => setDocumentFile(null)}>Eltávolítás</button>
            </div>
          )}
          <div className="actions">
            <button className="btn" onClick={handleUpload} disabled={disabledUpload}>Feltöltés és indexelés</button>
          </div>
          {uploadProgress > 0 && uploadProgress < 100 && (
            <div className="progress">
              <div className="progress-bar" style={{ width: `${uploadProgress}%` }} />
            </div>
          )}
        </section>

        <section className="card">
          <h2>Kérdés feltevése</h2>
          <div className="input-row">
            <textarea
              className="input"
              rows={4}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Pl.: Milyen témák szerepelnek a 3. fejezetben?"
              onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canAsk) handleQuery(); }}
            />
            <button className="btn" onClick={handleQuery} disabled={!canAsk}>Kérdés beküldése</button>
          </div>
          <div className="helper">Először tölts fel egy dokumentumot, majd tedd fel a kérdésed.</div>
        </section>

        {response && (
          <section className="card">
            <div className="card-header">
              <h2>Válasz</h2>
              <div className="card-actions">
                <button className="btn btn-secondary" onClick={copyAnswer}>Másolás</button>
              </div>
            </div>
            <div className="answer">
              <div className="answer-row"><span className="label">Válasz</span><div className="value">{response.answer}</div></div>
            </div>
            <div className="sources">
              <h3>Források</h3>
              <ul>
                {response.sources.map((source, idx) => (
                  <li key={idx}>
                    <details>
                      <summary>{source.section ? `${source.section}` : `Bekezdés ${idx + 1}`}</summary>
                      <pre className="source-pre">{source.content || source}</pre>
                    </details>
                  </li>
                ))}
              </ul>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">❤️-val készült — Kvíz RAG alkalmazás</footer>
    </div>
  );
}

export default App;