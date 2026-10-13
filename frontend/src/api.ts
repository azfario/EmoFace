import type {Analysis,AnalysisOptions,Enrollment,ModelInfo,UIStatus} from './types';
const messages:Record<string,string>={
  API_UNAVAILABLE:'The local service is offline or unreachable. Restart the local demo, then refresh this page.',
  OPERATOR_AUTH_REQUIRED:'Your session ended. Sign in again to continue.',
  ENROLLMENT_DISABLED:'The local operator workspace is disabled. Complete the local setup guide first.',
  ENROLLMENT_UNAVAILABLE:'The enrollment database or protected storage is unavailable. Check local setup.',
  RATE_LIMITED:'The local request limit was reached. Wait a minute before trying again.',
  CSRF_REQUIRED:'Your session needs a fresh sign-in.',
  CONSENT_REQUIRED:'Confirm adult consent and voluntary research participation first.',
  ENROLLMENT_CONSENT_REQUIRED:'Confirm permission to retain an encrypted reference.',
  MULTIPLE_FACES:'More than one face was detected. Use a frame with only the consenting adult.',
  LOW_QUALITY_IMAGE:'Keep one sufficiently large, fully visible face in adequate lighting. Enrollment and identity matching also require a clear photo.',
  NO_FACE:'No usable face was detected. Face the camera and try again.',
  INVALID_IMAGE:'Choose a valid JPEG or PNG image.',UNSUPPORTED_MEDIA_TYPE:'Only JPEG and PNG images are supported.',
  PAYLOAD_TOO_LARGE:'The image is too large. Use a file below 8 MiB and 10 million pixels.',
  DUPLICATE_IMAGE:'Choose distinct reference photos rather than repeated copies.',
  INCONSISTENT_ENROLLMENT:'The reference photos are inconsistent. Use clear photos of the same consenting adult.',
  DUPLICATE_DISPLAY_NAME:'That display name already exists. Select its record to re-enroll.',
  CUDA_UNAVAILABLE:'CUDA is unavailable. Processing is paused; CPU fallback is disabled.',
  MODEL_NOT_READY:'The required model is not ready. Check the local service status.',
  SERVICE_BUSY:'Another request is finishing. Wait a moment and try again.',
  ENROLLMENT_BUSY:'Another local operation is finishing. Wait a moment and try again.',
  INFERENCE_TIMEOUT:'The request timed out. Submitted work finishes local cleanup before another request can start.',
  ENROLLMENT_KEY_UNAVAILABLE:'The protected reference key is unavailable. Contact the local operator.',
  CANDIDATE_GALLERY_UNAVAILABLE:'The protected gallery cannot be read. No partial result was shown.',
  FACE_ANALYSIS_DISABLED:'Visible-expression analysis is disabled in the local service.',
  CANDIDATES_DISABLED:'Candidate matching is disabled in the local service.',
  ENROLLMENT_CLEANUP_PENDING:'Cleanup is pending. The record remains blocked; use the local recovery guide.',
  BLACK_FRAME_REJECTED:'The camera delivered a black frame. Check the selected camera source.',
};
export class PublicError extends Error {constructor(public code:string,public status=0){super(messages[code]??'The local request could not be completed. Check the service and try again.');}}
export class API {
  private csrf:string|null=null;
  private requests=new Set<AbortController>();
  constructor(private authLost:()=>void=()=>{}){}
  clear(){this.csrf=null;this.abort();}
  abort(){for(const request of this.requests)request.abort();this.requests.clear();}
  async request<T>(path:string,options:RequestInit={},signal?:AbortSignal):Promise<T>{
    if(!path.startsWith('/api/')&&!path.startsWith('/health'))throw new PublicError('INVALID_REQUEST');
    const controller=new AbortController();if(!options.keepalive)this.requests.add(controller);
    const abort=()=>controller.abort();signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)abort();
    const deadline=window.setTimeout(abort,90000);
    try{
      const response=await fetch(path,{...options,credentials:'same-origin',cache:'no-store',signal:controller.signal});
      const value=await response.json();
      if(!response.ok){if(response.status===401){this.csrf=null;this.authLost();}throw new PublicError(typeof value.status==='string'?value.status:'REQUEST_FAILED',response.status);}
      return value as T;
    }catch(error){if(error instanceof PublicError||error instanceof DOMException&&error.name==='AbortError')throw error;throw new PublicError('API_UNAVAILABLE');}
    finally{clearTimeout(deadline);this.requests.delete(controller);signal?.removeEventListener('abort',abort);}
  }
  mutation(extra:Record<string,string>={}){if(!this.csrf)throw new PublicError('OPERATOR_AUTH_REQUIRED',401);return {'x-faceattend-csrf':this.csrf,...extra};}
  async login(password:string){const value=await this.request<{csrf_token:string;expires_in_seconds:number}>('/api/v1/demo/session',{method:'POST',headers:{'Content-Type':'application/json','x-faceattend-login':'1'},body:JSON.stringify({password})});this.csrf=value.csrf_token;return value.expires_in_seconds;}
  async open(){const value=await this.request<{csrf_token:string;expires_in_seconds:number}>('/api/v1/demo/session',{method:'POST',headers:{'Content-Type':'application/json','x-faceattend-login':'1'},body:'{}'});this.csrf=value.csrf_token;return value.expires_in_seconds;}
  async logout(){const token=this.csrf;this.clear();if(token)await this.request('/api/v1/demo/session',{method:'DELETE',headers:{'x-faceattend-csrf':token},keepalive:true});}
  async status(){const [models,ui]=await Promise.all([this.request<ModelInfo>('/api/v1/models/info'),this.request<UIStatus>('/api/v1/demo/ui-status')]);return {models,ui};}
  list(page=1){return this.request<{enrollments:Enrollment[]}>('/api/v1/demo/enrollments?page='+page);}
  async enroll(name:string,photos:File[],days:number,identity?:string){
    if(photos.length<3||photos.length>5)throw new PublicError('INVALID_REQUEST');
    const body=new FormData();body.set('display_name',name);body.set('adult_consent','true');body.set('research_opt_in','true');body.set('enrollment_consent','true');body.set('retention_days',String(days));
    photos.forEach((photo,index)=>body.append('enrollment_images',photo,`reference-${index+1}.${photo.type==='image/png'?'png':'jpg'}`));
    const path='/api/v1/demo/enrollments'+(identity?'/'+encodeURIComponent(identity)+'/reenroll':'');
    return this.request<{identity_id:string}>(path,{method:'POST',headers:this.mutation(),body});
  }
  manage(identity:string,operation:'revoke'|'delete'){return this.request('/api/v1/demo/enrollments/'+encodeURIComponent(identity)+(operation==='revoke'?'/revoke':''),{method:operation==='delete'?'DELETE':'POST',headers:this.mutation()});}
  async analyze(image:Blob,options:AnalysisOptions,signal?:AbortSignal){
    const body=new FormData();body.set('adult_consent','true');body.set('research_opt_in','true');body.append('image',image,'voluntary-frame.jpg');
    const headers:Record<string,string>={};if(options.expression){headers['x-faceattend-expression-opt-in']='1';headers['x-faceattend-blur-opt-in']='1';}if(options.candidates)headers['x-faceattend-candidate-opt-in']='1';if(options.outline)headers['x-faceattend-box-opt-in']='1';
    const params=new URLSearchParams({include_candidates:String(options.candidates),include_face_box:String(options.outline),top_k:'3'});
    if(options.liveCamera){headers['x-faceattend-live-opt-in']='1';params.set('live_camera','true');}
    if(options.expression)params.set('allow_blurry_expression','true');
    const endpoint=options.expression?'analyze-face':'preview-face';
    const value=await this.request<Analysis>('/api/v1/demo/'+endpoint+'?'+params,{method:'POST',headers:this.mutation(headers),body},signal);
    if(value.verified!==false||value.threshold!==null||value.face_count<0||value.face_count>1||!Array.isArray(value.faces)||value.faces.length!==value.face_count)throw new PublicError('UNSAFE_RESULT');
    for(const face of value.faces){if(face.identity_diagnostic.verified!==false||face.identity_diagnostic.threshold!==null||face.identity_diagnostic.possible_candidates.some(c=>c.verified!==false)||!options.candidates&&face.identity_diagnostic.possible_candidates.length>0||!options.outline&&face.face_box)throw new PublicError('UNSAFE_RESULT');
      if(face.input_quality?.status==='BLUR_GATE_BYPASSED'&&face.identity_diagnostic.possible_candidates.length>0)throw new PublicError('UNSAFE_RESULT');
      const expression=face.visible_expression_prediction;if(expression&&(!options.expression||expression.score_calibrated!==false||expression.not_a_mood_or_diagnosis!==true||expression.domain_accuracy_validated!==false||!['surprise','fear','disgust','happy','sad','angry','neutral'].includes(expression.label)||!Number.isFinite(expression.softmax_score)||expression.softmax_score<0||expression.softmax_score>1))throw new PublicError('UNSAFE_RESULT');}
    return value;
  }
}
