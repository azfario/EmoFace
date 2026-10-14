export class Camera {
  stream:MediaStream|null=null;
  starting=false;
  private generation=0;
  constructor(private changed:(stream:MediaStream|null,error?:string,starting?:boolean)=>void=()=>{}){}
  async start(deviceId?:string){
    this.stop();const generation=this.generation;
    this.starting=true;this.changed(null,undefined,true);
    try{
      if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera access needs a secure localhost browser.');
      const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:deviceId?{deviceId:{exact:deviceId}}:{facingMode:'user'}});
      if(generation!==this.generation||document.visibilityState==='hidden'){stream.getTracks().forEach(track=>track.stop());return;}
      this.starting=false;this.stream=stream;for(const track of stream.getTracks())track.addEventListener('ended',()=>{if(this.stream===stream)this.stop('Camera access ended. Choose Start camera to reconnect.');},{once:true});
      this.changed(stream);
    }catch(error){if(generation!==this.generation)return;this.starting=false;const code=error instanceof DOMException?error.name:'';this.changed(null,code==='NotAllowedError'?'Camera permission was denied. You can choose an image instead.':code==='NotFoundError'?'No camera was found. Connect a camera or choose an image.':'The camera could not start. Check the selected source and browser permission.');}
  }
  stop(error?:string){this.generation++;this.starting=false;const stream=this.stream;this.stream=null;stream?.getTracks().forEach(track=>track.stop());this.changed(null,error);}
}
export function capture(video:HTMLVideoElement):Promise<Blob>{
  if(!video.srcObject||video.readyState<2||video.videoWidth<=0||video.videoHeight<=0)throw new Error('Wait for a live camera preview before capturing.');
  if(video.videoWidth*video.videoHeight>10000000)throw new Error('Choose a lower camera resolution before capturing.');
  const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
  const context=canvas.getContext('2d');if(!context)throw new Error('Frame capture is unavailable.');context.drawImage(video,0,0);
  return new Promise((resolve,reject)=>canvas.toBlob(blob=>{canvas.width=0;canvas.height=0;if(blob&&blob.size<=8*1024*1024)resolve(blob);else reject(new Error('The frame could not be captured within the image limit.'));},'image/jpeg',0.95));
}
export type AnalysisLimits={intervalMs:number;durationMs:number;maxFrames:number};
export class BoundedAnalysis {
  private timeout:ReturnType<typeof setTimeout>|undefined;
  private end:ReturnType<typeof setTimeout>|undefined;
  private generation=0;
  private controller:AbortController|null=null;
  active=false;
  constructor(private task:(signal:AbortSignal)=>Promise<void>,private changed:(active:boolean)=>void=()=>{},private limits:AnalysisLimits={intervalMs:10000,durationMs:45000,maxFrames:5},private finished:(reason:'limit'|'error')=>void=()=>{}){
    if((limits.intervalMs!==0&&limits.intervalMs<5000)||limits.durationMs<=0||limits.durationMs>300000||limits.maxFrames<1||limits.maxFrames>(limits.intervalMs===0?6000:60))throw new Error('Invalid analysis sampling limits.');
  }
  start(){this.stop();const generation=this.generation;let count=0;this.active=true;this.changed(true);this.end=setTimeout(()=>this.finish('limit'),this.limits.durationMs);
    const tick=async()=>{if(!this.active||generation!==this.generation)return;const started=performance.now();this.controller=new AbortController();try{await this.task(this.controller.signal);}catch{if(generation===this.generation)this.finish('error');return;}
      if(generation!==this.generation||!this.active)return;count++;if(count>=this.limits.maxFrames){this.finish('limit');return;}
      // Live mode has no post-inference pause; only cap fast responses at 20 fps.
      this.timeout=setTimeout(tick,this.limits.intervalMs===0?Math.max(0,50-(performance.now()-started)):this.limits.intervalMs);};void tick();}
  stop(){this.generation++;this.active=false;if(this.timeout)clearTimeout(this.timeout);if(this.end)clearTimeout(this.end);this.controller?.abort();this.controller=null;this.changed(false);}
  private finish(reason:'limit'|'error'){this.stop();this.finished(reason);}
}
