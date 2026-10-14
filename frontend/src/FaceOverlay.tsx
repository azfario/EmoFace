import {useId} from 'react';
import type {Analysis} from './types';
export function FaceOverlay({result}: {result: Analysis | null | undefined}) {
  const id = useId();
  const face = result?.faces[0], box = face?.face_box;
  if (!box || !result?.face_count) return null;
  const blurry = face.input_quality?.status === 'BLUR_GATE_BYPASSED';
  const candidate = blurry ? undefined : face.identity_diagnostic.possible_candidates[0];
  const fullName = candidate ? `Possible: ${candidate.display_name}` : 'Name: Unknown';
  const name = fullName.length > 34 ? fullName.slice(0, 33) + '…' : fullName;
  const emotion = face.visible_expression_prediction?.label;
  const unit = Math.min(box.image_width, box.image_height) / 200;
  const width = Math.min(194 * unit, box.image_width), height = (blurry ? 49 : 36) * unit;
  const x = Math.max(0, Math.min(box.x, box.image_width - width));
  const y = Math.max(0, Math.min(box.y - height, box.image_height - height));
  return <svg className="face-outline" viewBox={`0 0 ${box.image_width} ${box.image_height}`} preserveAspectRatio="xMidYMid meet" aria-label="Face predictions" role="img">
    <rect x={box.x} y={box.y} width={box.width} height={box.height} rx={4 * unit} fill="none" stroke="#d7f2a1" strokeWidth={1.5 * unit}/>
    <defs><clipPath id={id}><rect width={width} height={height} rx={3 * unit}/></clipPath></defs>
    <g className="face-caption" transform={`translate(${x} ${y})`} clipPath={`url(#${id})`}>
      <title>{fullName}</title><rect width={width} height={height} rx={3 * unit} fill="#13291f" fillOpacity="0.95"/>
      <text x={6 * unit} y={14 * unit} fill="#d7f2a1" fontSize={9 * unit} fontWeight="600">{name}</text>
      <text x={6 * unit} y={28 * unit} fill="#fff" fontSize={11 * unit} fontWeight="600">Emotion: {emotion ?? 'Unavailable'}</text>
      {blurry && <text x={6 * unit} y={42 * unit} fill="#ffdb93" fontSize={8 * unit}>Blurry</text>}
    </g>
  </svg>;
}
