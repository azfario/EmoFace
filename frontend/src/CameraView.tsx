import {useEffect, useRef, useState} from 'react';
import {Camera} from './camera';
import {FaceOverlay} from './FaceOverlay';
import type {Analysis} from './types';

export function ErrorNote({text}: {text: string}) { return text ? <div role="alert" className="notice error">{text}</div> : null; }
export function useBlobURL(blob: Blob | null) {
  const [url, setURL] = useState('');
  useEffect(() => {
    if (!blob) { setURL(''); return; }
    const value = URL.createObjectURL(blob); setURL(value);
    return () => URL.revokeObjectURL(value);
  }, [blob]);
  return url;
}
export function useCamera() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [device, setDevice] = useState('');
  const video = useRef<HTMLVideoElement>(null);
  const ended = useRef<() => void>(() => {});
  const alive = useRef(true);
  const refreshRef = useRef<() => Promise<void>>(async () => {});
  const [camera] = useState(() => new Camera((value, message, pending) => {
    if (!alive.current) return;
    setStream(value); setStarting(!!pending); setError(message ?? '');
    if (!value) ended.current(); else void refreshRef.current();
  }));
  refreshRef.current = async () => {
    try {
      const values = await navigator.mediaDevices?.enumerateDevices?.();
      if (alive.current && values) setDevices(values.filter(value => value.kind === 'videoinput'));
    } catch { /* Camera access reports actionable permission errors separately. */ }
  };
  useEffect(() => {
    alive.current = true;
    const refresh = () => void refreshRef.current();
    const hidden = () => { if (document.visibilityState === 'hidden') camera.stop(); };
    refresh(); navigator.mediaDevices?.addEventListener?.('devicechange', refresh);
    document.addEventListener('visibilitychange', hidden);
    return () => {
      alive.current = false; camera.stop();
      navigator.mediaDevices?.removeEventListener?.('devicechange', refresh);
      document.removeEventListener('visibilitychange', hidden);
    };
  }, [camera]);
  return {camera, stream, error, starting, devices, device, setDevice, video, ended};
}
export type CameraState = ReturnType<typeof useCamera>;
export function CameraSelector({media, disabled = false}: {media: CameraState; disabled?: boolean}) {
  return <label className="camera-source">Camera<select aria-label="Camera" value={media.device} disabled={disabled || !!media.stream || media.starting} onChange={event => media.setDevice(event.target.value)}>
    <option value="">Default camera</option>{media.devices.map((item, index) => <option key={item.deviceId} value={item.deviceId}>{item.label || `Camera ${index + 1}`}</option>)}
  </select></label>;
}
export function CameraPreview({media, result, onReady}: {media: CameraState; result?: Analysis | null; onReady?: () => void}) {
  useEffect(() => {
    const video = media.video.current;
    if (!video) return;
    video.srcObject = media.stream;
    if (media.stream) void video.play().catch(() => {});
    return () => { video.srcObject = null; };
  }, [media.stream, media.video]);
  function ready() { if (media.stream && media.video.current?.srcObject === media.stream && media.video.current.readyState >= 2) onReady?.(); }
  return <div className={media.stream ? 'preview live' : 'preview'}>
    <video ref={media.video} muted playsInline aria-label="Live camera feed" onLoadedData={ready} onPlaying={ready}/>
    {!media.stream && <div className="camera-placeholder"><span className="camera-icon" aria-hidden="true">◎</span>{media.starting ? 'Opening camera…' : 'Camera off'}</div>}
    {media.stream && <span className="live-badge"><span className="dot ready"/>Live</span>}
    {media.stream && result && result.face_count === 0 && <div role="status" className="frame-status">No face</div>}
    {media.stream && <FaceOverlay result={result}/>}
  </div>;
}
export function CameraButtons({media, start, stop, disabled = false}: {media: CameraState; start?: () => void; stop?: () => void; disabled?: boolean}) {
  return <div className="button-row">
    <button type="button" className="button primary" disabled={disabled || !!media.stream || media.starting} onClick={start ?? (() => void media.camera.start(media.device || undefined))}>Start</button>
    <button type="button" className="button" disabled={!media.stream && !media.starting} onClick={stop ?? (() => media.camera.stop())}>Stop</button>
  </div>;
}
