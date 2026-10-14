import {useEffect, useRef, useState} from 'react';
import {API} from './api';
import {BoundedAnalysis, capture} from './camera';
import {CameraButtons, CameraPreview, CameraSelector, ErrorNote, useCamera} from './CameraView';
import type {Analysis, ModelInfo, UIStatus} from './types';

export function Scan({api, status}: {api: API; status: {models: ModelInfo; ui: UIStatus}}) {
  const media = useCamera();
  const [result, setResult] = useState<Analysis | null>(null);
  const [error, setError] = useState('');
  const [requested, setRequested] = useState(false);
  const [ready, setReady] = useState(false);
  const alive = useRef(true);
  const epoch = useRef(0);
  const inFlight = useRef(false);
  const task = useRef<(signal: AbortSignal) => Promise<void>>(async () => {});
  const finished = useRef<(reason: 'limit' | 'error') => void>(() => {});
  const [loop] = useState(() => new BoundedAnalysis(signal => task.current(signal), () => {},
    {intervalMs: 0, durationMs: 300000, maxFrames: 6000}, reason => finished.current(reason)));
  const available = status.models.model_readiness.face === 'ready' && status.models.model_readiness.expression === 'ready'
    && status.ui.analysis_enabled && status.ui.candidates_enabled && ['ready', 'fictional'].includes(status.ui.registration_database);
  function cancel() {
    epoch.current++; loop.stop();
    if (alive.current) { setResult(null); setReady(false); setRequested(false); }
  }
  function stop() { cancel(); media.camera.stop(); }
  media.ended.current = cancel;
  finished.current = reason => { stop(); if (alive.current && reason === 'limit') setError('Scan finished. Press Start to continue.'); };
  async function start() {
    setError(''); stop(); await media.camera.start(media.device || undefined);
    if (alive.current && media.camera.stream && document.visibilityState !== 'hidden') setRequested(true);
  }
  task.current = async signal => {
    if (inFlight.current) return;
    const generation = epoch.current; inFlight.current = true;
    try {
      if (!media.video.current) throw new Error('Camera unavailable.');
      const image = await capture(media.video.current);
      if (signal.aborted || generation !== epoch.current) return;
      const value = await api.analyze(image, {expression: true, candidates: true, outline: true, liveCamera: true}, signal);
      if (alive.current && !signal.aborted && generation === epoch.current) setResult(value);
    } catch (caught) {
      if (alive.current && !signal.aborted && generation === epoch.current) setError(caught instanceof Error ? caught.message : 'Scan failed.');
      throw caught;
    } finally { inFlight.current = false; }
  };
  useEffect(() => {
    if (!requested || !ready || !media.stream) return;
    loop.start(); return () => loop.stop();
  }, [requested, ready, media.stream, loop]);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; epoch.current++; loop.stop(); api.abort(); };
  }, [api, loop]);
  return <section className="panel scan-panel">
    <div className="section-heading"><div><h1>Scan</h1><p className="caption">Predictions</p></div><CameraSelector media={media}/></div>
    <CameraPreview media={media} result={result} onReady={() => setReady(true)}/>
    <div className="scan-controls"><CameraButtons media={media} start={() => void start()} stop={stop} disabled={!available}/><span className="hint">Start confirms adult permission to scan.</span></div>
    <ErrorNote text={error || media.error || (!available ? 'Scanning unavailable. Check models and enrollment service.' : '')}/>
  </section>;
}
