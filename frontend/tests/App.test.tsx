import React from 'react';
import {it, expect, vi} from 'vitest';
import {render, screen, fireEvent, waitFor, act, within} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {App} from '../src/App';

const identity = {identity_status: 'Unknown', candidate_count: 1, possible_candidates: [{identity_id: 'fictional', display_name: 'Alex', label: 'Possible match', cosine_similarity: .8, verified: false}], verified: false, threshold: null, reason: 'THRESHOLD_NOT_CALIBRATED'};
const prediction = () => ({face_count: 1, face_detected: true, verified: false, threshold: null, faces: [{face_index: 0, identity_diagnostic: identity, visible_expression_prediction: {label: 'happy', softmax_score: .8, score_calibrated: false, not_a_mood_or_diagnosis: true, domain_accuracy_validated: false}, face_box: {x: 25, y: 60, width: 150, height: 130, image_width: 200, image_height: 200}}]});
const record = {identity_id: '12345678-1234-4234-8234-123456789abc', display_name: 'Alex', status: 'active', expires_at: '2026-11-01T00:00:00Z'};
const response = (value: unknown, status = 200) => new Response(JSON.stringify(value), {status});

function mock({loginRequired = false, rows = [] as typeof record[], analyze = async () => response(prediction())} = {}) {
  const fetch = vi.fn(async (path: string, options: RequestInit = {}) => {
    if (path.includes('models/info')) return response({gpu_device: 'Fixture GPU', model_readiness: {face: 'ready', expression: 'ready'}});
    if (path.includes('ui-status')) return response({fixture_mode: false, registration_database: 'ready', operator_auth_ready: true, login_required: loginRequired, candidates_enabled: true, analysis_enabled: true, telemetry_enabled: false, anonymous_totals: null});
    if (path.endsWith('/session')) return response(options.method === 'POST' ? {csrf_token: 'fictional-csrf', expires_in_seconds: 900} : {});
    if (path.includes('analyze-face')) return analyze();
    if (path.includes('/enrollments')) {
      if (options.method === 'DELETE') rows = [];
      return response(options.method ? {identity_id: record.identity_id, status: 'completed'} : {enrollments: rows});
    }
    return response({});
  });
  vi.stubGlobal('fetch', fetch);
  const stop = vi.fn();
  const ended = vi.fn();
  const stream = {getTracks: () => [{stop, addEventListener: ended}]} as unknown as MediaStream;
  const getUserMedia = vi.fn().mockResolvedValue(stream);
  Object.defineProperty(navigator, 'mediaDevices', {value: {getUserMedia, enumerateDevices: vi.fn().mockResolvedValue([{kind: 'videoinput', deviceId: 'fixture-camera', label: 'Fixture camera'}])}, configurable: true});
  return {fetch, stop, stream, getUserMedia, ended};
}
async function connected() { await screen.findByText('Connected'); }
async function start() { await userEvent.click(screen.getByRole('button', {name: 'Start'})); await screen.findByText('Live'); }
function ready() {
  const video = screen.getByLabelText('Live camera feed') as HTMLVideoElement;
  Object.defineProperties(video, {readyState: {value: 2, configurable: true}, videoWidth: {value: 200, configurable: true}, videoHeight: {value: 200, configurable: true}});
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({drawImage: vi.fn()} as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(done => done(new Blob(['fixture image'], {type: 'image/jpeg'})));
  return video;
}
async function enrollment() { await connected(); await userEvent.click(screen.getByRole('button', {name: 'Enroll'})); await screen.findByText('Saved names'); }
const files = (count: number) => Array.from({length: count}, (_, i) => new File(['fictional photo ' + i], 'private-name-' + i + '.png', {type: 'image/png'}));

it('highlights a failed reference, retains inputs, and clears the error when replacing it', async () => {
  const {fetch} = mock(); const original = fetch.getMockImplementation()!; let saves = 0;
  fetch.mockImplementation(async (path, options = {}) => path.endsWith('/enrollments') && options.method === 'POST' && ++saves === 1
    ? response({status: 'LOW_QUALITY_IMAGE', photo_number: 2, quality_reason: 'blur', message: 'private detail'}, 422) : original(path, options));
  render(<App/>); await enrollment(); await userEvent.type(screen.getByLabelText('Name'), 'Alex');
  await userEvent.upload(screen.getByLabelText('Upload reference photos'), files(3)); await userEvent.click(screen.getByRole('button', {name: 'Save'}));
  await screen.findByText('Photo 2 is too blurry. Replace it.');
  expect(screen.getByAltText('Reference 2').parentElement?.classList.contains('rejected')).toBe(true);
  expect(screen.getByAltText('Reference 2').parentElement?.getAttribute('aria-describedby')).toBe('enrollment-error');
  expect(screen.getAllByAltText(/Reference /)).toHaveLength(3); expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Alex');
  expect(screen.getByAltText('Reference 1').parentElement?.classList.contains('rejected')).toBe(false);
  await userEvent.click(screen.getByRole('button', {name: 'Remove reference 2'}));
  expect(screen.queryByRole('alert')).toBeNull(); expect(screen.queryByText('Replace')).toBeNull();
  await userEvent.upload(screen.getByLabelText('Upload reference photos'), files(1)); await userEvent.click(screen.getByRole('button', {name: 'Save'}));
  await screen.findByText('Saved.'); expect(screen.queryByAltText('Reference 1')).toBeNull();
});

it('opens on Scan with two tabs, no checkboxes, and no camera or browser storage access', async () => {
  const {getUserMedia} = mock(); const storage = vi.spyOn(Storage.prototype, 'setItem'); render(<App/>); await connected();
  expect(screen.getByRole('heading', {name: 'Scan'})).toBeTruthy(); expect(screen.getByRole('navigation').querySelectorAll('button')).toHaveLength(2);
  expect(screen.queryByRole('checkbox')).toBeNull(); expect(screen.queryByText('Overview')).toBeNull(); expect(getUserMedia).not.toHaveBeenCalled(); expect(storage).not.toHaveBeenCalled();
});
it('Start automatically requests names, emotion and outlines and renders both on the feed', async () => {
  const {fetch} = mock(); render(<App/>); await connected(); await start(); const video = ready(); fireEvent.loadedData(video);
  const overlay = await screen.findByRole('img', {name: 'Face predictions'}); expect(within(overlay).getByText('Possible: Alex', {selector: 'text'})).toBeTruthy(); expect(within(overlay).getByText('Emotion: happy')).toBeTruthy();
  const call = fetch.mock.calls.find(([path]) => path.includes('analyze-face'))!;
  expect(call[0]).toContain('include_candidates=true'); expect(call[0]).toContain('include_face_box=true'); expect(call[0]).toContain('live_camera=true');
  expect(call[1]!.headers).toMatchObject({'x-faceattend-csrf': 'fictional-csrf', 'x-faceattend-expression-opt-in': '1', 'x-faceattend-candidate-opt-in': '1'});
  expect((call[1]!.body as FormData).get('adult_consent')).toBe('true'); await userEvent.click(screen.getByRole('button', {name: 'Stop'}));
});
it('Stop releases tracks, clears labels and permits another explicit Start', async () => {
  const {stop, getUserMedia} = mock(); render(<App/>); await connected(); await start(); fireEvent.loadedData(ready()); await screen.findByText('Emotion: happy');
  await userEvent.click(screen.getByRole('button', {name: 'Stop'})); expect(stop).toHaveBeenCalledOnce(); expect(screen.queryByText('Emotion: happy')).toBeNull();
  await start(); expect(getUserMedia).toHaveBeenCalledTimes(2); expect(screen.queryByText('Emotion: happy')).toBeNull();
});
it('waits for inference and discards a late result after Stop', async () => {
  let resolve!: (value: Response) => void;
  const {fetch} = mock({analyze: () => new Promise<Response>(done => {resolve = done;})}); render(<App/>); await connected(); await start(); fireEvent.loadedData(ready());
  await waitFor(() => expect(fetch.mock.calls.filter(([path]) => path.includes('analyze-face'))).toHaveLength(1));
  await new Promise(done => setTimeout(done, 100)); expect(fetch.mock.calls.filter(([path]) => path.includes('analyze-face'))).toHaveLength(1);
  const signal = fetch.mock.calls.find(([path]) => path.includes('analyze-face'))![1]!.signal!;
  await userEvent.click(screen.getByRole('button', {name: 'Stop'})); expect(signal.aborted).toBe(true);
  await act(async () => resolve(response(prediction()))); expect(screen.queryByRole('img', {name: 'Face predictions'})).toBeNull();
});
it('no-face frames clear the previous name and emotion while scanning continues', async () => {
  let calls = 0; mock({analyze: async () => response(++calls === 1 ? prediction() : {face_count: 0, face_detected: false, verified: false, threshold: null, faces: []})});
  render(<App/>); await connected(); await start(); fireEvent.loadedData(ready()); await screen.findByText('No face'); expect(screen.queryByText('Emotion: happy')).toBeNull(); expect(screen.getByText('Live')).toBeTruthy();
});
it('service errors stop the camera and enable Start', async () => {
  const {stop} = mock({analyze: async () => response({status: 'CUDA_UNAVAILABLE'}, 503)}); render(<App/>); await connected(); await start(); fireEvent.loadedData(ready());
  await screen.findByRole('alert'); expect(stop).toHaveBeenCalledOnce(); expect((screen.getByRole('button', {name: 'Start'}) as HTMLButtonElement).disabled).toBe(false); expect(screen.queryByText('Live')).toBeNull();
});
it.each([
  ['NO_FACE', 'No face'], ['LOW_QUALITY_IMAGE', 'No clear face'], ['MULTIPLE_FACES', 'One face at a time'],
  ['BLACK_FRAME_REJECTED', 'Waiting for camera image'], ['INVALID_EXPRESSION_CROP', 'No clear face'],
])('keeps the camera live and continues analyzing after %s', async (code, message) => {
  const {stop, fetch, getUserMedia} = mock({analyze: async () => response({status: code}, code === 'BLACK_FRAME_REJECTED' ? 400 : 422)});
  render(<App/>); await connected(); await start(); fireEvent.loadedData(ready()); await screen.findByText(message);
  await waitFor(() => expect(fetch.mock.calls.filter(([path]) => path.includes('analyze-face')).length).toBeGreaterThanOrEqual(2));
  expect(stop).not.toHaveBeenCalled(); expect(getUserMedia).toHaveBeenCalledOnce(); expect(screen.getByText('Live')).toBeTruthy();
  expect(screen.queryByRole('alert')).toBeNull(); expect((screen.getByRole('button', {name: 'Stop'}) as HTMLButtonElement).disabled).toBe(false);
  await userEvent.click(screen.getByRole('button', {name: 'Stop'})); expect(stop).toHaveBeenCalledOnce(); expect(screen.queryByText(message)).toBeNull();
});
it('clears stale predictions on a rejected frame and recovers without reopening the camera', async () => {
  let calls = 0, resume = false;
  const {stop, getUserMedia} = mock({analyze: async () => response(++calls === 1 || resume ? prediction() : {status: 'LOW_QUALITY_IMAGE'}, calls === 1 || resume ? 200 : 422)});
  render(<App/>); await connected(); await start(); fireEvent.loadedData(ready()); await screen.findByText('Emotion: happy');
  await screen.findByText('No clear face'); expect(screen.queryByText('Emotion: happy')).toBeNull(); expect(screen.queryByRole('img', {name: 'Face predictions'})).toBeNull();
  resume = true; await screen.findByText('Emotion: happy'); expect(screen.queryByText('No clear face')).toBeNull();
  expect(stop).not.toHaveBeenCalled(); expect(getUserMedia).toHaveBeenCalledOnce(); await userEvent.click(screen.getByRole('button', {name: 'Stop'}));
});
it('permission denial gives a retry message without analyzing', async () => {
  const {getUserMedia, fetch} = mock(); getUserMedia.mockRejectedValue(new DOMException('private message', 'NotAllowedError')); render(<App/>); await connected(); await userEvent.click(screen.getByRole('button', {name: 'Start'}));
  await screen.findByText(/Camera permission was denied/); expect(fetch.mock.calls.some(([path]) => path.includes('analyze-face'))).toBe(false);
});
it('Stop releases a camera permission result arriving later', async () => {
  const {getUserMedia, stream, stop} = mock(); let resolve!: (value: MediaStream) => void; getUserMedia.mockImplementation(() => new Promise(done => {resolve = done;}));
  render(<App/>); await connected(); await userEvent.click(screen.getByRole('button', {name: 'Start'})); await screen.findByText('Opening camera…'); await userEvent.click(screen.getByRole('button', {name: 'Stop'}));
  await act(async () => resolve(stream)); expect(stop).toHaveBeenCalledOnce(); expect(screen.queryByText('Live')).toBeNull();
});
it('switching to Enroll releases the scan camera and clears results', async () => {
  const {stop} = mock(); render(<App/>); await connected(); await start(); fireEvent.loadedData(ready()); await screen.findByText('Emotion: happy'); await userEvent.click(screen.getByRole('button', {name: 'Enroll'}));
  expect(stop).toHaveBeenCalledOnce(); expect(screen.queryByText('Emotion: happy')).toBeNull(); expect(screen.queryByRole('checkbox')).toBeNull();
});
it('hidden tabs release the camera and reconnect without restarting it', async () => {
  const {stop, getUserMedia} = mock(); render(<App/>); await connected(); await start();
  Object.defineProperty(document, 'visibilityState', {value: 'hidden', configurable: true}); await act(async () => document.dispatchEvent(new Event('visibilitychange'))); expect(stop).toHaveBeenCalledOnce();
  Object.defineProperty(document, 'visibilityState', {value: 'visible', configurable: true}); await act(async () => document.dispatchEvent(new Event('visibilitychange'))); await connected(); expect(getUserMedia).toHaveBeenCalledOnce(); expect(screen.getByText('Camera off')).toBeTruthy();
});
it('five-minute limit stops camera even with a pending request', async () => {
  const {stop, fetch} = mock({analyze: () => new Promise<Response>(() => {})}); render(<App/>); await connected(); await start(); const video = ready(); vi.useFakeTimers();
  await act(async () => { fireEvent.loadedData(video); });
  await act(async () => { await vi.advanceTimersByTimeAsync(300000); });
  expect(stop).toHaveBeenCalledOnce(); expect(screen.getByText('Scan finished. Press Start to continue.')).toBeTruthy(); expect(fetch.mock.calls.filter(([path]) => path.includes('analyze-face'))).toHaveLength(1);
});
it('session expiry releases the camera and never restarts it', async () => {
  const {fetch, stop, getUserMedia} = mock(); const original = fetch.getMockImplementation()!;
  fetch.mockImplementation(async (path, options = {}) => path.endsWith('/session') && options.method === 'POST' ? response({csrf_token: 'fictional-csrf', expires_in_seconds: 1}) : original(path, options));
  render(<App/>); await connected(); await start(); await waitFor(() => expect(stop).toHaveBeenCalledOnce(), {timeout: 2000}); expect(getUserMedia).toHaveBeenCalledOnce();
});
it('selected camera is used and revoked tracks stop scanning', async () => {
  const {getUserMedia, ended, stop} = mock(); render(<App/>); await connected(); await screen.findByRole('option', {name: 'Fixture camera'}); await userEvent.selectOptions(screen.getByLabelText('Camera'), 'fixture-camera'); await start();
  expect(getUserMedia).toHaveBeenCalledWith({audio: false, video: {deviceId: {exact: 'fixture-camera'}}}); await act(async () => ended.mock.calls[0][1]()); expect(stop).toHaveBeenCalledOnce(); expect(screen.queryByText('Live')).toBeNull();
});
it('upload enrollment needs a name and three photos, then saves without checkboxes', async () => {
  const {fetch} = mock(); render(<App/>); await enrollment(); const save = screen.getByRole('button', {name: 'Save'}) as HTMLButtonElement; expect(save.disabled).toBe(true);
  await userEvent.type(screen.getByLabelText('Name'), 'Alex'); await userEvent.upload(screen.getByLabelText('Upload reference photos'), files(2)); expect(save.disabled).toBe(true);
  await userEvent.upload(screen.getByLabelText('Upload reference photos'), files(1)); expect(save.disabled).toBe(false); await userEvent.click(save); await screen.findByText('Saved.');
  const body = fetch.mock.calls.find(([path, options]) => path.endsWith('/enrollments') && options?.method === 'POST')![1]!.body as FormData;
  expect(body.get('enrollment_consent')).toBe('true'); expect(body.get('retention_days')).toBe('30'); expect(body.getAll('enrollment_images')).toHaveLength(3); expect(screen.queryByRole('checkbox')).toBeNull(); expect(screen.queryByAltText('Reference 1')).toBeNull();
});
it('manual capture adds a photo and releases its preview URL on removal', async () => {
  const revoke = vi.spyOn(URL, 'revokeObjectURL'); mock(); render(<App/>); await enrollment(); await start(); ready(); await userEvent.click(screen.getByRole('button', {name: 'Capture'})); await screen.findByAltText('Reference 1');
  await userEvent.click(screen.getByRole('button', {name: 'Remove reference 1'})); expect(revoke).toHaveBeenCalled(); expect(screen.getByText('0/5 photos')).toBeTruthy();
});
it('Stop discards a pending manual capture', async () => {
  mock(); render(<App/>); await enrollment(); await start(); ready(); let done!: BlobCallback;
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(callback => {done = callback;}); await userEvent.click(screen.getByRole('button', {name: 'Capture'})); await userEvent.click(screen.getByRole('button', {name: 'Stop'}));
  await act(async () => done(new Blob(['late'], {type: 'image/jpeg'}))); expect(screen.queryByAltText('Reference 1')).toBeNull();
});
it('rejects excessive or unsupported reference photos', async () => {
  mock(); render(<App/>); await enrollment(); await userEvent.upload(screen.getByLabelText('Upload reference photos'), files(6)); await screen.findByText('Maximum five photos.'); expect(screen.queryByAltText('Reference 1')).toBeNull();
  fireEvent.change(screen.getByLabelText('Upload reference photos'), {target: {files: [new File(['bad'], 'input.txt', {type: 'text/plain'})]}}); await screen.findByText('Use JPEG or PNG photos under 8 MiB.');
});
it('re-enrollment retains the name and requires fresh photos', async () => {
  mock({rows: [record]}); render(<App/>); await enrollment(); await screen.findByText('Alex'); await userEvent.click(screen.getByRole('button', {name: 'Re-enroll'}));
  expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Alex'); expect((screen.getByLabelText('Name') as HTMLInputElement).readOnly).toBe(true); expect((screen.getByRole('button', {name: 'Save'}) as HTMLButtonElement).disabled).toBe(true);
});
it('name validation supports modern browser pattern syntax', async () => {
  mock(); render(<App/>); await enrollment();
  const input = screen.getByLabelText('Name') as HTMLInputElement;
  const pattern = new RegExp('^(?:' + input.pattern + ')$', 'v');
  expect(pattern.test('Alex-Smith')).toBe(true); expect(pattern.test('Alex Smith')).toBe(true); expect(pattern.test('Alex/Smith')).toBe(false);
});
it('deletion is confirmed and keyboard Escape cancels the dialog', async () => {
  const {fetch} = mock({rows: [record]}); render(<App/>); await enrollment(); await screen.findByText('Alex'); await userEvent.click(screen.getByRole('button', {name: 'Delete'})); await userEvent.keyboard('{Escape}'); expect(screen.queryByRole('dialog')).toBeNull();
  expect(fetch.mock.calls.some(([path, options]) => path.includes(record.identity_id) && options?.method === 'DELETE')).toBe(false);
  await userEvent.click(screen.getByRole('button', {name: 'Delete'})); await userEvent.click(screen.getByRole('button', {name: 'Confirm delete'})); await screen.findByText('Deleted.'); await screen.findByText('No saved names');
});
it('optional password mode remains available and clears submitted passwords', async () => {
  mock({loginRequired: true}); render(<App/>); await screen.findByLabelText('Password'); await userEvent.type(screen.getByLabelText('Password'), 'fictional-password'); await userEvent.click(screen.getByRole('button', {name: 'Sign in'})); await connected(); expect(screen.queryByLabelText('Password')).toBeNull(); expect(screen.getByRole('button', {name: 'Sign out'})).toBeTruthy();
});
