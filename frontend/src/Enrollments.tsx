import React, {useCallback, useEffect, useRef, useState} from 'react';
import {API, PublicError} from './api';
import {capture} from './camera';
import {CameraButtons, CameraPreview, CameraSelector, ErrorNote, useBlobURL, useCamera} from './CameraView';
import {useModal} from './modal';
import type {Enrollment} from './types';

function Photo({file, index, disabled, invalid, remove}: {file: File; index: number; disabled: boolean; invalid: boolean; remove: () => void}) {
  const url = useBlobURL(file);
  return <div className={invalid ? 'photo rejected' : 'photo'} aria-describedby={invalid ? 'enrollment-error' : undefined}><img src={url || undefined} alt={`Reference ${index + 1}`}/><button type="button" disabled={disabled} aria-label={`Remove reference ${index + 1}`} onClick={remove}>×</button>{invalid && <span className="photo-rejection">Replace</span>}</div>;
}

export function Enrollments({api}: {api: API}) {
  const media = useCamera();
  const [records, setRecords] = useState<Enrollment[]>([]);
  const [page, setPage] = useState(1);
  const [name, setName] = useState('');
  const [photos, setPhotos] = useState<File[]>([]);
  const [identity, setIdentity] = useState<string | undefined>();
  const [busy, setBusy] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState('');
  const [rejectedPhoto, setRejectedPhoto] = useState<number | undefined>();
  const [message, setMessage] = useState('');
  const [confirm, setConfirm] = useState<{record: Enrollment; action: 'revoke' | 'delete'} | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const alive = useRef(true);
  const captureEpoch = useRef(0);
  const captureBusy = useRef(false);
  const loadEpoch = useRef(0);
  const photosRef = useRef<File[]>([]);
  const locked = busy || capturing;
  media.ended.current = () => { captureEpoch.current++; };
  const dialog = useModal(!!confirm, () => { if (!busy) setConfirm(null); });

  function changePhotos(value: File[]) { photosRef.current = value; setPhotos(value); setRejectedPhoto(undefined); setError(''); }
  const load = useCallback(async () => {
    const generation = ++loadEpoch.current;
    try {
      const value = await api.list(page);
      if (alive.current && generation === loadEpoch.current) setRecords(value.enrollments);
    } catch (caught) { if (alive.current && generation === loadEpoch.current && !(caught instanceof DOMException)) setError(caught instanceof Error ? caught.message : 'Gallery unavailable.'); }
  }, [api, page]);
  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; captureEpoch.current++; loadEpoch.current++; api.abort(); };
  }, [api]);

  function reset() {
    captureEpoch.current++; media.camera.stop(); changePhotos([]); setName(''); setIdentity(undefined);
    if (input.current) input.current.value = '';
  }
  function add(files: File[]) {
    if (!files.length) return;
    if (files.some(file => !['image/jpeg', 'image/png'].includes(file.type) || file.size > 8 * 1024 * 1024)) { setError('Use JPEG or PNG photos under 8 MiB.'); return; }
    if (photosRef.current.length + files.length > 5) { setError('Maximum five photos.'); return; }
    changePhotos([...photosRef.current, ...files]); setError(''); setMessage('');
  }
  async function capturePhoto() {
    if (captureBusy.current || busy || !media.video.current) return;
    captureBusy.current = true; setCapturing(true);
    const generation = captureEpoch.current;
    try {
      const blob = await capture(media.video.current);
      if (!alive.current || generation !== captureEpoch.current || !media.camera.stream) return;
      add([new File([blob], `reference-${photosRef.current.length + 1}.jpg`, {type: 'image/jpeg'})]);
    } catch (caught) { if (alive.current && generation === captureEpoch.current) setError(caught instanceof Error ? caught.message : 'Capture failed.'); }
    finally { captureBusy.current = false; if (alive.current) setCapturing(false); }
  }
  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (locked || !name || photos.length < 3 || photos.length > 5) return;
    setBusy(true); setError(''); setRejectedPhoto(undefined); setMessage(''); media.camera.stop();
    try {
      await api.enroll(name, photos, 30, identity);
      if (!alive.current) return;
      reset(); setMessage('Saved.'); await load();
    } catch (caught) { if (alive.current && !(caught instanceof DOMException)) { setError(caught instanceof Error ? caught.message : 'Save failed.'); setRejectedPhoto(caught instanceof PublicError ? caught.photoNumber : undefined); } }
    finally { if (alive.current) setBusy(false); }
  }
  async function manage() {
    if (!confirm || locked) return;
    setBusy(true); setError(''); setMessage('');
    try {
      await api.manage(confirm.record.identity_id, confirm.action);
      if (!alive.current) return;
      setConfirm(null); setMessage(confirm.action === 'delete' ? 'Deleted.' : 'Revoked.'); await load();
    } catch (caught) { if (alive.current && !(caught instanceof DOMException)) setError(caught instanceof Error ? caught.message : 'Update failed.'); }
    finally { if (alive.current) setBusy(false); }
  }
  return <div className="enroll-layout">
    <section className="panel">
      <div className="section-heading"><h1>{identity ? 'Re-enroll' : 'Enroll'}</h1><CameraSelector media={media} disabled={locked}/></div>
      <form onSubmit={event => void save(event)}>
        <label className="field">Name<input value={name} onChange={event => setName(event.target.value)} maxLength={64} pattern={'[A-Za-z0-9][A-Za-z0-9 _.:\\-]{0,63}'} readOnly={!!identity} disabled={locked} required autoComplete="off" placeholder="Name or nickname"/></label>
        <CameraPreview media={media}/>
        <div className="enroll-controls"><CameraButtons media={media} disabled={locked}/><button type="button" className="button utility" disabled={locked || !media.stream || photos.length >= 5} onClick={() => void capturePhoto()}>{capturing ? 'Capturing…' : 'Capture'}</button><button type="button" className="button upload-button" disabled={locked || photos.length >= 5} onClick={() => { media.camera.stop(); input.current?.click(); }}>Upload photos</button></div>
        <input ref={input} className="file-input" type="file" tabIndex={-1} multiple accept="image/jpeg,image/png" aria-label="Upload reference photos" disabled={locked} onChange={event => { add(Array.from(event.target.files ?? [])); event.target.value = ''; }}/>
        <div className="photos-heading"><span>{photos.length}/5 photos</span><span className="hint">3–5 distinct photos</span></div>
        <div className="photo-grid">{photos.map((file, index) => <Photo key={index} file={file} index={index} disabled={locked} invalid={rejectedPhoto === index + 1} remove={() => changePhotos(photosRef.current.filter((_, i) => i !== index))}/>)}{Array.from({length: 5 - photos.length}, (_, index) => <div className="photo-slot" aria-hidden="true" key={`empty-${index}`}><span>{photos.length + index + 1}</span></div>)}</div>
        <div className="save-row"><button className="button primary" disabled={locked || photos.length < 3 || !name}>{busy ? 'Saving…' : 'Save'}</button>{identity && <button type="button" className="button" disabled={locked} onClick={reset}>Cancel</button>}</div>
        <p className="hint permission-note">Start confirms adult permission to capture. Save authorizes encrypted storage for 30 days.</p>
        <ErrorNote id="enrollment-error" text={error || media.error}/>{message && <div role="status" className="notice success">{message}</div>}
      </form>
    </section>
    <section className="panel gallery">
      <div className="section-heading"><h2>Saved names</h2><span className="caption">{page}/4</span></div>
      {!records.length && <div className="empty">No saved names</div>}
      {records.map(record => <article className="record" key={record.identity_id}>
        <div className="record-heading"><span className="avatar">{record.display_name.charAt(0).toUpperCase()}</span><div><h3>{record.display_name}</h3><span className="caption">{record.status.replaceAll('_', ' ')} · {new Date(record.expires_at).toLocaleDateString()}</span></div></div>
        <div className="record-actions"><button className="text-button" disabled={locked || record.status === 'pending' || record.status === 'deletion_pending'} onClick={() => { reset(); setIdentity(record.identity_id); setName(record.display_name); setError(''); setMessage(''); }}>Re-enroll</button><button className="text-button" disabled={locked || record.status !== 'active'} onClick={() => setConfirm({record, action: 'revoke'})}>Revoke</button><button className="text-button danger" disabled={locked} onClick={() => setConfirm({record, action: 'delete'})}>Delete</button></div>
      </article>)}
      {(page > 1 || records.length === 25) && <div className="pagination"><button className="button" disabled={locked || page === 1} onClick={() => setPage(value => value - 1)}>Previous</button><button className="button" disabled={locked || page === 4 || records.length < 25} onClick={() => setPage(value => value + 1)}>Next</button></div>}
    </section>
    {confirm && <div className="modal-backdrop"><section ref={dialog} className="modal panel" role="dialog" aria-modal="true" aria-labelledby="confirm-title"><h2 id="confirm-title">{confirm.action === 'delete' ? 'Delete this name?' : 'Revoke this reference?'}</h2><p>{confirm.record.display_name}</p><div className="button-row"><button className={confirm.action === 'delete' ? 'button danger' : 'button primary'} disabled={busy} onClick={() => void manage()}>Confirm {confirm.action}</button><button className="button" disabled={busy} onClick={() => setConfirm(null)}>Cancel</button></div></section></div>}
  </div>;
}
