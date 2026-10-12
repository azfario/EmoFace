import type {Analysis} from './types';

export function FaceOverlay({result}:{result:Analysis|null|undefined}){
  const face=result?.faces[0],box=face?.face_box;if(!box)return null;
  const expression=face.visible_expression_prediction;
  const label=expression?`Emotion: ${expression.label}`:'Face detected';
  const unit=Math.min(box.image_width,box.image_height)/200;
  const width=Math.min(180*unit,box.image_width),height=expression?34*unit:21*unit;
  const x=Math.max(0,Math.min(box.x,box.image_width-width)),y=Math.max(0,box.y-height);
  return <svg className="face-outline" viewBox={`0 0 ${box.image_width} ${box.image_height}`} aria-label="Opt-in face outline" role="img">
    <rect x={box.x} y={box.y} width={box.width} height={box.height} rx={4*unit} fill="none" stroke="#d7f2a1" strokeWidth={1.5*unit}/>
    <g className="face-caption" transform={`translate(${x} ${y})`}>
      <rect width={width} height={height} rx={3*unit} fill="#13291f" fillOpacity="0.95"/>
      <text x={6*unit} y={14*unit} fill="#d7f2a1" fontSize={12*unit} fontWeight="700">{label}</text>
      {expression&&<text x={6*unit} y={27*unit} fill="#fff" fontSize={8*unit}>{face.input_quality?.status==='BLUR_GATE_BYPASSED'?'Uncalibrated - blurry input':'Uncalibrated prediction'}</text>}
    </g>
  </svg>;
}
