(function(){const a=document.createElement("link").relList;if(a&&a.supports&&a.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))i(r);new MutationObserver(r=>{for(const o of r)if(o.type==="childList")for(const u of o.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&i(u)}).observe(document,{childList:!0,subtree:!0});function e(r){const o={};return r.integrity&&(o.integrity=r.integrity),r.referrerPolicy&&(o.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?o.credentials="include":r.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function i(r){if(r.ep)return;r.ep=!0;const o=e(r);fetch(r.href,o)}})();const ne=`struct Camera { viewProj : mat4x4<f32> }
@group(0) @binding(0) var<uniform> uCamera : Camera;
struct VSIn {
  @location(0) position : vec4<f32>
}

struct VSOut {
  @builtin(position) position : vec4<f32>
}

@vertex
fn vs_main(in_ : VSIn) -> VSOut {
  var out : VSOut;
  let pos = in_.position.xyz;
  out.position = uCamera.viewProj * vec4<f32>(pos, 1.0);
  return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`,re=`struct Params {
  dims : vec2<u32>,
  pad0 : vec2<u32>,
  originSpacing : vec4<f32>,
  heightNoise : vec4<f32>
}

@group(0) @binding(0) var<uniform> uParams : Params;
@group(0) @binding(1) var<storage, read_write> positions : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> indirect : array<u32>;

fn ridge(v: f32) -> f32 {
  let r = 1.0 - abs(v);
  return r * r;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= uParams.dims.x || gid.y >= uParams.dims.y) {
    return;
  }

  let idx = gid.y * uParams.dims.x + gid.x;
  let spacing = uParams.originSpacing.zw;
  let offset = uParams.originSpacing.xy;
  let dimsF = vec2<f32>(uParams.dims);
  let center = (vec2<f32>(vec2<u32>(gid.xy)) / max(dimsF - vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 1.0))) - 0.5;
  let posXZ = (vec2<f32>(vec2<u32>(gid.xy)) * spacing) + offset;

  let time = uParams.heightNoise.w;
  let basePos = vec3<f32>(posXZ.x * uParams.heightNoise.y, posXZ.y * uParams.heightNoise.y, time);
  var height = fbm(basePos) * uParams.heightNoise.x;

  let ridged = ridge(fbm(basePos * 0.6) * 2.0 - 1.0) * uParams.heightNoise.z;
  height = height + ridged;

  let mask = max(0.0, 1.0 - length(center) * 1.6);
  height = height + mask * uParams.heightNoise.x * 0.6;

  positions[idx] = vec4<f32>(posXZ.x, height, posXZ.y, 1.0);

  if (gid.x == 0u && gid.y == 0u) {
    indirect[0] = uParams.dims.x * uParams.dims.y;
    indirect[1] = 1u;
    indirect[2] = 0u;
    indirect[3] = 0u;
  }
}

`,ie=`fn hash3(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453123);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }

fn noise3(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let n000 = hash3(i + vec3<f32>(0.0,0.0,0.0));
  let n100 = hash3(i + vec3<f32>(1.0,0.0,0.0));
  let n010 = hash3(i + vec3<f32>(0.0,1.0,0.0));
  let n110 = hash3(i + vec3<f32>(1.0,1.0,0.0));
  let n001 = hash3(i + vec3<f32>(0.0,0.0,1.0));
  let n101 = hash3(i + vec3<f32>(1.0,0.0,1.0));
  let n011 = hash3(i + vec3<f32>(0.0,1.0,1.0));
  let n111 = hash3(i + vec3<f32>(1.0,1.0,1.0));
  let nx00 = lerp(n000, n100, u.x);
  let nx10 = lerp(n010, n110, u.x);
  let nx01 = lerp(n001, n101, u.x);
  let nx11 = lerp(n011, n111, u.x);
  let nxy0 = lerp(nx00, nx10, u.y);
  let nxy1 = lerp(nx01, nx11, u.y);
  return lerp(nxy0, nxy1, u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
  var a = 0.5;
  var f = 1.0;
  var s = 0.0;
  for (var i = 0; i < 5; i = i + 1) {
    s = s + a * noise3(p * f);
    f = f * 2.0;
    a = a * 0.5;
  }
  return s;
}

`;function oe(n,a,e,i){const r=1/Math.tan(n/2),o=1/(e-i),u=new Float32Array(16);return u[0]=r/a,u[5]=r,u[10]=(i+e)*o,u[11]=-1,u[14]=2*i*e*o,u}function ae(n,a,e){const[i,r,o]=n,[u,P,T]=a;let g=i-u,y=r-P,v=o-T,f=Math.hypot(g,y,v);g/=f,y/=f,v/=f;let U=e[1]*v-e[2]*y,S=e[2]*g-e[0]*v,b=e[0]*y-e[1]*g;f=Math.hypot(U,S,b),U/=f,S/=f,b/=f;const A=y*b-v*S,R=v*U-g*b,B=g*S-y*U,c=new Float32Array(16);return c[0]=U,c[1]=A,c[2]=g,c[3]=0,c[4]=S,c[5]=R,c[6]=y,c[7]=0,c[8]=b,c[9]=B,c[10]=v,c[11]=0,c[12]=-(U*i+S*r+b*o),c[13]=-(A*i+R*r+B*o),c[14]=-(g*i+y*r+v*o),c[15]=1,c}function se(n,a){const e=new Float32Array(16);for(let i=0;i<4;i++){const r=n[i],o=n[i+4],u=n[i+8],P=n[i+12];e[i]=r*a[0]+o*a[1]+u*a[2]+P*a[3],e[i+4]=r*a[4]+o*a[5]+u*a[6]+P*a[7],e[i+8]=r*a[8]+o*a[9]+u*a[10]+P*a[11],e[i+12]=r*a[12]+o*a[13]+u*a[14]+P*a[15]}return e}const z=document.getElementById("app"),d=document.createElement("canvas");d.width=z.clientWidth;d.height=z.clientHeight;z.appendChild(d);const V=[],_={log:console.log,warn:console.warn,error:console.error};function m(n){V.push(`[${new Date().toISOString()}] ${n}`)}console.log=(...n)=>{_.log.apply(console,n),m(n.map(String).join(" "))};console.warn=(...n)=>{_.warn.apply(console,n),m("WARN "+n.map(String).join(" "))};console.error=(...n)=>{_.error.apply(console,n),m("ERROR "+n.map(String).join(" "))};async function ce(){const n=V.join(`
`);if("showSaveFilePicker"in window)try{const o=await(await window.showSaveFilePicker({suggestedName:"webgpu-log.txt",types:[{accept:{"text/plain":[".txt"]}}]})).createWritable();await o.write(n),await o.close();return}catch{}const a=new Blob([n],{type:"text/plain"}),e=URL.createObjectURL(a),i=document.createElement("a");i.href=e,i.download="webgpu-log.txt",i.click(),URL.revokeObjectURL(e)}function ue(){const n=document.createElement("div");n.style.position="absolute",n.style.top="8px",n.style.left="8px",n.style.display="flex",n.style.gap="8px";const a=document.createElement("button"),e=document.createElement("button");a.textContent="Select Log Dir",e.textContent="Save Log";let i=null;a.onclick=async()=>{try{i=await window.showDirectoryPicker(),m("Selected log directory")}catch{m("Dir select canceled")}},e.onclick=async()=>{if(i)try{const o=await(await i.getFileHandle("webgpu-log.txt",{create:!0})).createWritable();await o.write(V.join(`
`)),await o.close(),m("Saved log to chosen directory");return}catch{m("Failed writing to chosen directory, falling back")}ce()},n.appendChild(a),n.appendChild(e),z.appendChild(n)}function k(n){console.error(n)}async function le(){if(ue(),window.onerror=(t,s,l,w,h)=>{m(`window.onerror ${t} @${s}:${l}:${w} ${h?.stack||""}`)},window.addEventListener("unhandledrejection",t=>{m(`unhandledrejection ${String(t.reason)}`)}),!("gpu"in navigator))throw new Error("WebGPU not supported");const n=navigator.gpu,a=await n.requestAdapter();if(!a)throw new Error("No adapter");const e=await a.requestDevice();try{e.addEventListener&&e.addEventListener("uncapturederror",t=>{m(`device uncapturederror: ${t?.error?.message||t}`)})}catch{}const i=d.getContext("webgpu"),r=n.getPreferredCanvasFormat();i.configure({device:e,format:r,alphaMode:"opaque"});let o=e.createTexture({size:{width:d.width,height:d.height},format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT});function u(){const t=Math.max(1,Math.min(2,window.devicePixelRatio||1)),s=Math.floor(z.clientWidth*t),l=Math.floor(z.clientHeight*t);s===d.width&&l===d.height||(d.width=s,d.height=l,i.configure({device:e,format:r,alphaMode:"opaque"}),o.destroy(),o=e.createTexture({size:{width:s,height:l},format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}))}window.addEventListener("resize",u),u();const P=e.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),T=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]});e.pushErrorScope("validation");const g=e.createShaderModule({code:ne}),y=await g.getCompilationInfo();if(y.messages?.length)for(const t of y.messages)k(`terrain.wgsl: ${t.lineNum}:${t.linePos} ${t.message}`);const v=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[T]}),vertex:{module:g,entryPoint:"vs_main",buffers:[{arrayStride:16,attributes:[{shaderLocation:0,offset:0,format:"float32x4"}]}]},fragment:{module:g,entryPoint:"fs_main",targets:[{format:r}]},primitive:{topology:"point-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}});await e.popErrorScope().catch(t=>k(`Render pipeline error: ${String(t)}`));const f={x:256,z:256},U=f.x*f.z,S=e.createBuffer({size:U*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.VERTEX}),b=e.createBuffer({size:4*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT}),A=e.createBindGroup({layout:T,entries:[{binding:0,resource:{buffer:P}}]}),R=64,B=e.createBuffer({size:R,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),c=new ArrayBuffer(R),N=new Uint32Array(c),E=new Float32Array(c),F=.12;N[0]=f.x,N[1]=f.z,N[2]=0,N[3]=0,E[4]=-256*F*.5,E[5]=-256*F*.5,E[6]=F,E[7]=F,E[8]=3.2,E[9]=.35,E[10]=1.6,E[11]=0,e.queue.writeBuffer(B,0,c);const j=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),D=e.createShaderModule({code:ie+`
`+re});try{const t=await D.getCompilationInfo();for(const s of t.messages||[])m(`heightmap.wgsl: ${s.lineNum}:${s.linePos} ${s.message}`)}catch{}const X=e.createComputePipeline({layout:e.createPipelineLayout({bindGroupLayouts:[j]}),compute:{module:D,entryPoint:"main"}}),Z=e.createBindGroup({layout:j,entries:[{binding:0,resource:{buffer:B}},{binding:1,resource:{buffer:S}},{binding:2,resource:{buffer:b}}]}),p=new Set;window.addEventListener("keydown",t=>{t.repeat||p.add(t.code)}),window.addEventListener("keyup",t=>{p.delete(t.code)}),window.addEventListener("blur",()=>p.clear());const x=[0,3,12];let $=Math.PI,G=-.25,q=!1;d.addEventListener("click",()=>d.requestPointerLock()),document.addEventListener("pointerlockchange",()=>{q=document.pointerLockElement===d}),window.addEventListener("mousemove",t=>{if(!q)return;const s=.0025;$-=t.movementX*s,G-=t.movementY*s;const l=Math.PI/2-.05;G=Math.max(-l,Math.min(l,G))});function I(t){const s=Math.hypot(t[0],t[1],t[2]);return s<1e-5?[0,0,0]:[t[0]/s,t[1]/s,t[2]/s]}function W(t,s){return[t[1]*s[2]-t[2]*s[1],t[2]*s[0]-t[0]*s[2],t[0]*s[1]-t[1]*s[0]]}function O(t,s,l){t[0]+=s[0]*l,t[1]+=s[1]*l,t[2]+=s[2]*l}let H=performance.now();const Y=[0,1,0];function Q(t){const s=d.width/Math.max(1,d.height),l=oe(60*Math.PI/180,s,.1,400),w=I([Math.cos(G)*Math.sin($),Math.sin(G),Math.cos(G)*Math.cos($)]);let h=W(w,Y);Math.hypot(h[0],h[1],h[2])<1e-4&&(h=[1,0,0]),h=I(h);const L=I(W(h,w)),M=(p.has("ShiftLeft")||p.has("ShiftRight")?18:10)*t;p.has("KeyW")&&O(x,w,M),p.has("KeyS")&&O(x,w,-M),p.has("KeyA")&&O(x,h,-M),p.has("KeyD")&&O(x,h,M),(p.has("KeyE")||p.has("Space"))&&O(x,L,M),(p.has("KeyQ")||p.has("ControlLeft"))&&O(x,L,-M);const J=[x[0]+w[0],x[1]+w[1],x[2]+w[2]],ee=ae(x,J,L),te=se(ee,l);e.queue.writeBuffer(P,0,te)}async function K(){const t=performance.now(),s=Math.min(.1,(t-H)/1e3);H=t,Q(s),E[11]=t*5e-4,e.queue.writeBuffer(B,0,c);const l=e.createCommandEncoder();try{e.pushErrorScope("validation");const C=l.beginComputePass();C.setPipeline(X),C.setBindGroup(0,Z),C.dispatchWorkgroups(Math.ceil(f.x/8),Math.ceil(f.z/8),1),C.end();const M=await e.popErrorScope();M&&m(`Compute validation: ${M.message}`)}catch(C){k(`Compute pipeline error: ${String(C)}`)}const w=i.getCurrentTexture().createView(),h=o.createView(),L=l.beginRenderPass({colorAttachments:[{view:w,clearValue:{r:.02,g:.02,b:.03,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:h,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});L.setPipeline(v),L.setBindGroup(0,A),L.setVertexBuffer(0,S),L.drawIndirect(b,0),L.end(),e.queue.submit([l.finish()]),requestAnimationFrame(K)}requestAnimationFrame(K)}le();
