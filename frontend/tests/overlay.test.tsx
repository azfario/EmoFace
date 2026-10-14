import React from 'react';
import {it, expect} from 'vitest';
import {render, screen} from '@testing-library/react';
import {FaceOverlay} from '../src/FaceOverlay';
import type {Analysis} from '../src/types';
function result(y = 70, name?: string): Analysis {
  return {face_count: 1, face_detected: true, verified: false, threshold: null, faces: [{face_index: 0, identity_diagnostic: {identity_status: 'Unknown', candidate_count: name ? 1 : 0, possible_candidates: name ? [{identity_id: 'fixture', display_name: name, label: 'Possible match', cosine_similarity: .8, verified: false}] : [], verified: false, threshold: null, reason: 'diagnostic'}, face_box: {x: 25, y, width: 150, height: 120, image_width: 200, image_height: 200}, visible_expression_prediction: {label: 'neutral', softmax_score: .2, score_calibrated: false, not_a_mood_or_diagnosis: true, domain_accuracy_validated: false}}]};
}
it('places both highest-ranked name and emotion inside the original-frame overlay', () => {
  render(<FaceOverlay result={result(70, 'Alex')}/>); const svg = screen.getByRole('img', {name: 'Face predictions'});
  expect(svg.getAttribute('viewBox')).toBe('0 0 200 200'); expect(svg.getAttribute('preserveAspectRatio')).toBe('xMidYMid meet'); expect(svg.querySelector('g')?.getAttribute('transform')).toBe('translate(6 34)'); expect(screen.getByText('Possible: Alex', {selector: 'text'})).toBeTruthy(); expect(screen.getByText('Emotion: neutral')).toBeTruthy();
});
it('shows Unknown when no candidate exists', () => { render(<FaceOverlay result={result()}/>); expect(screen.getByText('Name: Unknown', {selector: 'text'})).toBeTruthy(); });
it('suppresses identity names on blurry frames', () => {
  const value = result(70, 'Alex'); value.faces[0].input_quality = {status: 'BLUR_GATE_BYPASSED', warning: 'Blurry'}; render(<FaceOverlay result={value}/>);
  expect(screen.queryByText('Possible: Alex')).toBeNull(); expect(screen.getByText('Name: Unknown', {selector: 'text'})).toBeTruthy(); expect(screen.getByText('Blurry')).toBeTruthy(); expect(screen.getByText('Emotion: neutral')).toBeTruthy();
});
it('clamps captions at the image edge and clears them on no face', () => {
  const {rerender} = render(<FaceOverlay result={result(10, 'Alex')}/>); expect(screen.getByRole('img').querySelector('g')?.getAttribute('transform')).toBe('translate(6 0)'); rerender(<FaceOverlay result={{...result(), face_count: 0, faces: []}}/>); expect(screen.queryByRole('img')).toBeNull();
});
it('truncates long names and clips the caption inside its bounds', () => {
  render(<FaceOverlay result={result(70, 'A'.repeat(64))}/>); const svg = screen.getByRole('img'); expect(svg.querySelector('text')?.textContent).toHaveLength(34); expect(svg.querySelector('text')?.textContent?.endsWith('…')).toBe(true); expect(svg.querySelector('clipPath')).toBeTruthy();
});
it('keeps only the first ranked name', () => {
  const value = result(70, 'Alex'); value.faces[0].identity_diagnostic.possible_candidates.push({identity_id: 'second', display_name: 'Sam', label: 'Possible match', cosine_similarity: .7, verified: false}); render(<FaceOverlay result={value}/>); expect(screen.queryByText(/Sam/)).toBeNull();
});
