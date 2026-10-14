import React, {useCallback, useEffect, useRef, useState} from 'react';
import {API} from './api';
import {Scan} from './Scan';
import {Enrollments} from './Enrollments';
import {ErrorNote} from './CameraView';
import type {Page, ModelInfo, UIStatus} from './types';

export function App() {
  const [page, setPage] = useState<Page>('scan');
  const [status, setStatus] = useState<{models: ModelInfo; ui: UIStatus} | null>(null);
  const [signed, setSigned] = useState(false);
  const [visible, setVisible] = useState(document.visibilityState !== 'hidden');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [retry, setRetry] = useState(0);
  const [ending, setEnding] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const alive = useRef(true);
  const resetRef = useRef<() => void>(() => {});
  const [api] = useState(() => new API(() => resetRef.current()));
  const loginFree = status?.ui.login_required === false;

  const reset = useCallback(() => {
    api.clear();
    if (timer.current) clearTimeout(timer.current);
    if (!alive.current) return;
    setSigned(false); setPassword(''); setPage('scan');
  }, [api]);
  resetRef.current = reset;
  const end = useCallback(async () => {
    setEnding(true);
    const pending = api.logout();
    reset();
    try { await pending; }
    catch (caught) { if (alive.current) setError(caught instanceof Error ? caught.message : 'Sign out failed.'); }
    finally { if (alive.current) setEnding(false); }
  }, [api, reset]);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; api.clear(); if (timer.current) clearTimeout(timer.current); };
  }, [api]);
  useEffect(() => {
    let current = true;
    void api.status().then(value => { if (current) { setStatus(value); setError(''); } })
      .catch(() => { if (current) setError('Service unavailable.'); });
    return () => { current = false; };
  }, [api, retry]);
  useEffect(() => {
    const hidden = () => {
      const showing = document.visibilityState !== 'hidden';
      setVisible(showing);
      if (!showing) { setPassword(''); if (signed) void end(); else api.abort(); }
    };
    document.addEventListener('visibilitychange', hidden);
    return () => document.removeEventListener('visibilitychange', hidden);
  }, [api, end, signed]);
  useEffect(() => {
    if (!loginFree || signed || !visible || ending) return;
    let current = true;
    setError('');
    void api.open().then(seconds => {
      if (!current) return;
      setSigned(true);
      timer.current = setTimeout(() => void end(), seconds * 1000);
    }).catch(caught => { if (current) setError(caught instanceof Error ? caught.message : 'Connection failed.'); });
    return () => { current = false; };
  }, [api, end, loginFree, signed, visible, ending, retry]);

  async function login(event: React.FormEvent) {
    event.preventDefault();
    const value = password;
    setPassword(''); setBusy(true); setError('');
    try {
      const seconds = await api.login(value);
      if (!alive.current || document.visibilityState === 'hidden') { api.clear(); return; }
      setSigned(true);
      timer.current = setTimeout(() => void end(), seconds * 1000);
    } catch (caught) { if (alive.current) setError(caught instanceof Error ? caught.message : 'Sign in failed.'); }
    finally { if (alive.current) setBusy(false); }
  }
  function navigate(next: Page) { if (next !== page) { api.abort(); setPage(next); } }

  return <div className="app-shell">
    <header className="app-header">
      <a className="brand" href="#" onClick={event => { event.preventDefault(); navigate('scan'); }}>EmoFace</a>
      <nav aria-label="Main tabs">
        <button className={page === 'scan' ? 'tab selected' : 'tab'} aria-current={page === 'scan' ? 'page' : undefined} onClick={() => navigate('scan')}>Scan</button>
        <button className={page === 'enroll' ? 'tab selected' : 'tab'} aria-current={page === 'enroll' ? 'page' : undefined} onClick={() => navigate('enroll')}>Enroll</button>
      </nav>
    </header>
    <div className="system-strip"><span className="system-label">Local camera studio</span><div className="connection"><span className={signed ? 'dot ready' : 'dot'}/>{signed ? 'Connected' : 'Connecting'}{signed && !loginFree && <button className="text-button" onClick={() => void end()}>Sign out</button>}</div></div>
    <main>
      {status?.ui.fixture_mode && <div className="notice">Test data</div>}
      <ErrorNote text={error}/>
      {error && <button className="button" onClick={() => setRetry(value => value + 1)}>Retry</button>}
      {!signed && status && !loginFree && <form className="panel login" onSubmit={event => void login(event)}>
        <h1>Sign in</h1><label className="field">Password<input type="password" value={password} onChange={event => setPassword(event.target.value)} maxLength={256} autoComplete="off" required/></label>
        <button className="button primary" disabled={busy || !password}>{busy ? 'Signing in…' : 'Sign in'}</button>
      </form>}
      {!signed && (!status || loginFree) && !error && <div className="empty">Connecting…</div>}
      {signed && page === 'scan' && <Scan api={api} status={status!}/>}
      {signed && page === 'enroll' && <Enrollments api={api}/>}
    </main>
  </div>;
}
