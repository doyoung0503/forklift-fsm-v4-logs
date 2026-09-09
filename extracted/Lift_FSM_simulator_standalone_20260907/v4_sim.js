"use strict";

const $ = id => document.getElementById(id);
const app = {info:null, session:null, frames:[], report:null, activeOptions:null, activeConfig:null, cursor:0, running:false,
  busy:false, batches:[], grids:[], cancel:false, recording:null, video:null, lastWall:0,externalWorld:null,
  lastFrame:null,view:{zoom:100,x:0,z:0}};
const palletImage=new Image();
palletImage.onload=()=>{if(app.displayFrame) draw(app.displayFrame);};
palletImage.onerror=()=>{$("palletImageStatus").textContent="팔레트 이미지 로드 실패 · 치수 외곽선 표시";};
palletImage.src=LiftViewGeometry.palletImage.src;
const fmt = (v,n=3) => v == null ? "—" : Number(v).toFixed(n);
const reasonText = r => r.success ? "삽입 검증 통과" : r.reasons.join(" / ");

async function api(route, data) {
  const response = await fetch(`/api/${route}`, data === undefined ? {} : {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(data)
  });
  let result;
  try { result = await response.json(); }
  catch (_) { throw new Error("Python 서버 응답이 아닙니다. run_v4_simulator.bat로 실행하세요."); }
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

function options() {
  return {inference_mode:"requested", ...Object.fromEntries([...document.querySelectorAll("[data-option]")].map(e => [e.id,
    e.tagName === "SELECT" ? e.value : Number(e.value)]))};
}
function overrides() {
  return Object.fromEntries([...document.querySelectorAll("[data-config]")]
    .filter(e => e.value.trim() !== "").map(e => [e.dataset.config,Number(e.value)]));
}
function setOptions(values) {
  Object.entries({placement_mode:"relative",forklift_x:0,forklift_z:0,forklift_heading:0,
    pallet_x:0,pallet_z:2,pallet_heading:180,...values}).forEach(([key,value]) => { if ($(key)) $(key).value = value; });
  placementControls();
}
function placementControls() {
  $("worldPlacement").hidden=$("placement_mode").value!=="world";
  $("relativePlacement").hidden=$("placement_mode").value!=="relative";
}
function setOverrides(values) {
  document.querySelectorAll("[data-config]").forEach(e => {e.value=values[e.dataset.config] ?? "";});
}
function controls() {
  ["runBtn","stepBtn","resetBtn","fastBtn","applyScenarioBtn","applyConstantsBtn","clearConstantsBtn",
   "runBatchBtn","runGridBtn","replayCheckBtn","downloadTrajectoryCsvBtn","downloadTrajectoryJsonBtn"]
    .forEach(id => $(id).disabled = app.busy || !app.info || !!app.externalWorld);
  $("runBtn").textContent = app.running ? "일시정지" : "실행";
  // Pause remains available while a camera frame/native ACK is in flight.
  $("runBtn").disabled = (!app.running && app.busy) || !app.info || !!app.externalWorld;
  $("downloadBatchBtn").disabled = !app.batches.length;
  $("downloadBatchJsonBtn").disabled = !app.batches.length && !app.grids.length;
}
async function toggleRun() {
  if(app.serverClock && !app.report) {
    const desired=!app.running;
    if(desired&&app.busy) return;
    try {
      await api(desired?'play':'pause',{id:app.session,rate:Number($("timeScale").value)});
      app.running=desired;app.lastPerfWall=null;app.lastWall=performance.now();controls();
    } catch(error) {$("connectionStatus").textContent=error.message;}
    return;
  }
  if(app.running) {app.running=false;controls();return;}
  if(app.busy) return;
  if(app.report&&app.cursor>=app.frames.length-1) {app.cursor=0;$("replaySlider").value="0";}
  app.running=true;app.lastPerfWall=null;app.lastWall=performance.now();controls();
}
async function exclusive(fn) {
  if (app.busy) return;
  app.running=false; app.busy=true; controls();
  try { if(app.serverClock&&app.session&&!app.report) await api('pause',{id:app.session}); await fn(); }
  catch (error) { $("connectionStatus").textContent=error.message; }
  finally {app.busy=false;controls();}
}
async function reset() {
  LiftCameraView.reset();
  const selectedOptions=options(),selectedOverrides=overrides();
  app.lastPerfWall=null;
  const result=await api("session",{options:selectedOptions,overrides:selectedOverrides,replace_id:app.session});
  if(result.info) applyInfo(result.info);
  app.activeOptions=selectedOptions;app.activeConfig={...app.info.config,...selectedOverrides};
  app.session=result.id;app.frames=[];app.report=null;app.cursor=0;
  app.serverClock=true;app.serverCursor=0;app.displaySamples=[];app.lastPerfWall=null;
  // Initial placement has no running clock to synchronize. Show it even while
  // the 3D renderer/native window is starting up.
  app.displayFrame=result.frame;
  draw(result.frame);
  await render(result.frame);
  $("resultReadout").textContent="실행 대기";
  $("resultReadout").className="result-box";
  $("connectionStatus").textContent=`연결됨`;
  $("replayCheckStatus").textContent="검증 대기";
  $("logBox").textContent="PRECHECK";
  $("replaySlider").max="0";$("replaySlider").value="0";
}
async function appendFrames(frames) {
  app.frames.push(...frames);app.cursor=app.frames.length-1;
  $("replaySlider").max=String(Math.max(0,app.frames.length-1));
  $("replaySlider").value=String(app.cursor);await render(frames.at(-1));
}
function showReport(report) {
  app.report=report;
  renderLogging(report.logging);
  $("resultReadout").textContent=`${reasonText(report)} · FSM ${report.state} · ${fmt(report.elapsed,2)} s · 실제 잔여 ${fmt(report.true_remaining)} m · FSM 잔여 ${fmt(report.fsm_remaining)} m`;
  $("resultReadout").className=`result-box ${report.success?"passed":"failed"}`;
}
async function loadReport(report,live=false) {
  LiftCameraView.reset();
  if(live&&report.provenance) applyInfo(report.provenance);
  app.running=false;app.session=live?(report.window_session_id||null):null;app.frames=report.frames;
  if(!app.session) {
    // A saved run gets a display-only session: no tick, CAN output or run log.
    const viewer=await api("session",{options:report.options,overrides:report.overrides||{},show_window:true});
    app.session=viewer.id;
  }
  app.activeOptions=report.options;app.activeConfig=report.config;
  app.cursor=Math.max(0,app.frames.length-1);
  $("replaySlider").max=String(app.cursor);$("replaySlider").value=String(app.cursor);
  showReport(report);
  if(app.frames.length) await render(app.frames.at(-1));
  controls();
  $("replayCheckStatus").textContent="선택된 실행의 원본 FSM 재입력 검증을 실행할 수 있습니다.";
}
async function step(count=1,duration=null) {
  if(app.report) {
    if(app.cursor>=app.frames.length-1) {app.running=false;controls();return;}
    if(duration!=null) {
      const until=app.frames[app.cursor].t+duration;
      do {app.cursor++;} while(app.cursor<app.frames.length-1&&app.frames[app.cursor].t<until);
    } else app.cursor=Math.min(app.frames.length-1,app.cursor+count);
    $("replaySlider").value=String(app.cursor);await render(app.frames[app.cursor]);return;
  }
  const started=performance.now(),simBefore=app.displayFrame?.t ?? 0;
  const result=await api("step",{id:app.session,count,...(duration==null?{}:{duration})});
  if(app.serverClock) app.serverCursor+=result.frames.length;
  const stepMs=performance.now()-started;
  app.lastRenderTiming=null;
  await appendFrames(result.frames);
  const finished=performance.now(),last=result.frames.at(-1);
  const sample={browser_mono_ms:finished,sim_time_s:last.t,state:last.state,command:last.command,
    sim_advance_s:last.t-simBefore,cycle_ms:finished-started,
    interval_ms:app.lastPerfWall==null?null:finished-app.lastPerfWall,
    time_scale:Number($("timeScale").value),document_hidden:document.hidden,
    frame_count:app.frames.length,step_request_ms:stepMs,server_step_ms:result.server_step_ms,
    controller_tick_ms:result.controller_tick_ms,
    ...app.lastRenderTiming};
  app.lastPerfWall=finished;
  try {await api('performance',{id:app.session,sample});}
  catch(error) {$("connectionStatus").textContent=`성능 로그 저장 실패: ${error.message}`;}
  if(result.done) {app.running=false;showReport(result.report);controls();}
}
async function fullReport() {
  if(app.report?.trace?.length) return app.report;
  if(!app.session) throw new Error("저장할 실행 데이터가 없습니다.");
  const report=await api("report",{id:app.session});
  if(!report.trace.length) throw new Error("한 프레임 이상 실행한 뒤 저장하세요.");
  return report;
}
function download(content,mime,name) {
  const url=URL.createObjectURL(new Blob([content],{type:mime}));
  const a=document.createElement("a");a.href=url;a.download=name;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const csv = rows => "\uFEFF"+rows.map(row=>row.map(v=>`"${String(v??"").replaceAll('"','""')}"`).join(",")).join("\r\n");

function renderLogging(log) {
  const label={waiting:'실행 시작 시 자동 저장',recording:'파일 기록 중',closed:'저장 완료',error:'로그 저장 오류',disabled:'자동 저장 꺼짐'};
  $("runLogStatus").textContent=log?`${label[log.status]||log.status}${log.error?' · '+log.error:''}`:'자동 저장 정보가 없는 실행 기록입니다.';
  $("runLogPath").textContent=log?.directory||'';
  const count=log?.counts;
  $("runLogCounts").textContent=count?`CAN ${count.can_frames??0} · 모델 결과 ${count.model_results??0} · FSM 입력 ${count.fsm_steps??0} · 상태 기록 ${count.transitions??0} · 실제 위치 ${count.relative_poses??0}`:'';
}
function render(frame) {
  if(!frame) return Promise.resolve(false);
  return LiftCameraView.update(frame,app.activeOptions||options(),(shown,ack,timing)=>{
    app.lastRenderTiming=timing;
    app.displayFrame=shown;app.lastFrame=shown;
    app.displayCursor=app.frames.indexOf(shown);
    paintFrame(shown,ack);
  },app.session,error=>{
    app.running=false;controls();
    $("connectionStatus").textContent=`화면 동기화 오류: ${error.message}`;
  });
}
function paintFrame(frame,ack) {
  renderLogging(app.report?.logging||frame.logging);
  $("simClock").textContent=`t = ${fmt(frame.t,2)} s`;
  $("replayClock").textContent=`${app.frames.length?(app.displayCursor??app.cursor)+1:0} / ${app.frames.length}`;
  draw(frame);
  $("simCanvas").dataset.simTime=String(frame.t);
  $("simClock").textContent=`t = ${fmt(frame.t,3)} s`;
  drawRecording(frame);
  const events=[];
  let last="";
  for(const f of app.frames.slice(0,(app.displayCursor??app.cursor)+1)) {
    const key=`${f.state}/${f.command}`;
    if(key!==last) {events.push(`${fmt(f.t,2)} s  ${key}\n${f.lines.join("\n")}`);last=key;}
  }
  $("logBox").textContent=events.slice(-60).join("\n\n") || frame.lines.join("\n");
}
function canvasContext(id) {
  const canvas=$(id),rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
  const w=Math.max(100,rect.width),h=Math.max(100,rect.height);
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)) {
    canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  }
  const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  return {ctx,w,h};
}
function draw(frame) {
  const {ctx,w,h}=canvasContext("simCanvas"),t=frame.truth;
  const config=app.report?.vehicle_profile?.parameters || app.info?.vehicle_profile?.parameters || app.activeConfig || {};
  const o=app.activeOptions || options();
  const v=LiftViewGeometry.viewport(w,h,app.view.zoom,app.view.x,app.view.z);
  const {xmin,xmax,zmin,zmax,side,left,top,scale}=v;
  const W=(x,z)=>[left+(x-xmin)*scale,top+(zmax-z)*scale];
  const P=(x,z)=>W(...LiftViewGeometry.worldPoint(t,config,x,z));
  $("viewExtent").textContent=`${app.view.zoom}% · ${fmt(v.span,1)} × ${fmt(v.span,1)} m · X ${fmt(xmin,1)}~${fmt(xmax,1)} / Z ${fmt(zmin,1)}~${fmt(zmax,1)}`;
  ctx.fillStyle="#f8fafc";ctx.fillRect(0,0,w,h);
  const line=(a,b,color,width=1)=>{ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();};
  const grid=v.span>20?5:v.span>5?1:.5;
  ctx.font="10px sans-serif";ctx.fillStyle="#68707d";
  for(let x=Math.ceil(xmin/grid)*grid;x<=xmax+1e-9;x+=grid) {
    line(W(x,zmin),W(x,zmax),Math.abs(x)<1e-9?"#9aa7b8":"#e3e8ef");
    ctx.textAlign="center";ctx.fillText(fmt(x,grid<1?1:0),Math.max(left+12,Math.min(left+side-12,W(x,zmin)[0])),top+side-6);
  }
  for(let z=Math.ceil(zmin/grid)*grid;z<=zmax+1e-9;z+=grid) {
    line(W(xmin,z),W(xmax,z),Math.abs(z)<1e-9?"#9aa7b8":"#e3e8ef");
    ctx.textAlign="left";ctx.fillText(fmt(z,grid<1?1:0),left+4,Math.max(top+12,Math.min(top+side-20,W(xmin,z)[1]+3)));
  }
  ctx.textAlign="left";ctx.strokeStyle="#9aa7b8";ctx.strokeRect(left,top,side,side);
  ctx.save();ctx.beginPath();ctx.rect(left,top,side,side);ctx.clip();
  const poly=(points,fill,stroke)=>{ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...P(...p)):ctx.moveTo(...P(...p)));ctx.closePath();ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();}};
  const fov=LiftViewGeometry.fovPolygon(v,LiftViewGeometry.worldPoint(t,config,0,0),t.heading,o.camera_hfov_deg||55);
  if(fov.length) {
    ctx.beginPath();fov.forEach((p,i)=>i?ctx.lineTo(...W(...p)):ctx.moveTo(...W(...p)));ctx.closePath();
    ctx.fillStyle="#2f6fed0c";ctx.fill();ctx.strokeStyle="#85ace9";ctx.lineWidth=1.5;ctx.stroke();
  }
  const dimensions=LiftViewGeometry.dimensions(config,o);
  const pallet=(x,z)=>LiftViewGeometry.palletPoint(t,x,z);
  const rect=(left,right,front,back,color,stroke)=>poly([pallet(left,front),pallet(right,front),pallet(right,back),pallet(left,back)],color,stroke);
  const width=dimensions.width/2,opening=dimensions.opening/2;
  if(palletImage.complete&&palletImage.naturalWidth) {
    ctx.save();
    ctx.transform(...LiftViewGeometry.imageTransform((x,z)=>P(...pallet(x,z)),2*width,o.pallet_depth));
    ctx.drawImage(palletImage,...LiftViewGeometry.palletImage.crop,0,0,1,1);
    ctx.restore();
    $("palletImageStatus").textContent="실물 상면 이미지 · 포크는 상판 위에 겹쳐 표시";
  }
  rect(-width,width,0,o.pallet_depth,"transparent","#287e78");
  const spans=o.opening==="split"?[[-.4,-.15],[.15,.4]]:[[-opening,opening]];
  for(const [l,r] of spans) {
    // A top photograph cannot describe the underside insertion channel.
    ctx.setLineDash([4,3]);rect(l,r,0,o.pallet_depth,"#ffffff18","#125e59");ctx.setLineDash([]);
    line(P(...pallet(l,0)),P(...pallet(r,0)),"#04a69b",3);
  }
  const arrow=(a,b,color)=>{
    line(a,b,color,2);
    const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);
    for(const d of [-.5,.5]) line(b,[b[0]-7*Math.cos(angle+d),b[1]-7*Math.sin(angle+d)],color,2);
  };
  arrow(P(...pallet(0,0)),P(...pallet(0,-.45)),"#287e78");
  poly([[-.5,-1.6],[.5,-1.6],[.5,-.1],[-.5,-.1]],"#dce3eb","#697586");
  const halfFork=dimensions.forkSpan/2;
  [[-halfFork,-halfFork+o.fork_width],[halfFork-o.fork_width,halfFork]].forEach(([l,r])=>
    poly([[l,0],[r,0],[r,dimensions.cameraToTip],[l,dimensions.cameraToTip]],frame.collision?"#e34a4a":"#e8b923","#786211"));
  ctx.fillStyle="#2f6fed";ctx.beginPath();ctx.arc(...P(0,0),4,0,2*Math.PI);ctx.fill();
  const pivotX=config.CAMERA_TO_ROT_CENTER_X_M??0;
  ctx.fillStyle="#374151";ctx.beginPath();ctx.arc(...P(pivotX,dimensions.pivotZ),4,0,2*Math.PI);ctx.fill();
  arrow(P(0,-.65),P(0,.35),"#187650");
  $("geometryReadout").textContent=`팔레트 ${fmt(dimensions.width,2)} × ${fmt(dimensions.depth,2)} m · ${o.opening==="split"?"대조 실험: 분리 포켓 0.25 m × 2":`삽입 외측 폭 ${fmt(dimensions.opening,2)} m`} · 포크 외측 폭 ${fmt(dimensions.forkSpan,2)} m · 카메라→포크 끝 ${fmt(dimensions.cameraToTip,2)} m · 회전 중심→카메라 ${fmt(-dimensions.pivotZ,2)} m · 회전 중심→포크 끝 ${fmt(dimensions.pivotToTip,2)} m`;
  if(app.view.zoom>=175) {
    const dimension=(a,b,label,color)=>{
      line(a,b,color);const angle=Math.atan2(b[1]-a[1],b[0]-a[0]);
      const nx=-Math.sin(angle)*4,ny=Math.cos(angle)*4;
      for(const p of [a,b]) line([p[0]-nx,p[1]-ny],[p[0]+nx,p[1]+ny],color);
      ctx.font="11px sans-serif";ctx.textAlign="center";
      const x=(a[0]+b[0])/2,y=(a[1]+b[1])/2-5;
      ctx.fillStyle="#ffffffed";ctx.fillRect(x-ctx.measureText(label).width/2-3,y-11,ctx.measureText(label).width+6,14);
      ctx.fillStyle=color;ctx.fillText(label,x,y);
    };
    dimension(P(...pallet(-width,o.pallet_depth+.12)),P(...pallet(width,o.pallet_depth+.12)),`${fmt(dimensions.width,2)} m`,"#125e59");
    dimension(P(...pallet(width+.16,0)),P(...pallet(width+.16,o.pallet_depth)),`${fmt(dimensions.depth,2)} m`,"#125e59");
    if(o.opening!=="split") dimension(P(...pallet(-opening,-.13)),P(...pallet(opening,-.13)),`${fmt(dimensions.opening,2)} m`,"#125e59");
    dimension(P(-halfFork,-.08),P(halfFork,-.08),`${fmt(dimensions.forkSpan,2)} m`,"#786211");
    dimension(P(-halfFork-.2,dimensions.pivotZ),P(-halfFork-.2,dimensions.cameraToTip),`${fmt(dimensions.pivotToTip,2)} m`,"#374151");
  }
  ctx.restore();
  ctx.font="12px sans-serif";ctx.fillStyle="#364152";
  $("worldPoseCaption").textContent=`지게차 X ${fmt(t.world_x,2)} / Z ${fmt(t.world_z,2)} m · 방향 ${fmt(t.heading,1)}° · ${o.camera_hfov_deg||55}° FOV`;
}
function values(text) {
  const entries=text.split(",").map(v=>v.trim());
  if(!entries.length||entries.some(v=>v===""||!Number.isFinite(Number(v)))) throw new Error("조건은 쉼표로 구분한 숫자로 입력하세요.");
  return [...new Set(entries.map(Number))];
}
function conditions() {
  const repeats=Number($("batchRepeats").value);
  if(!Number.isInteger(repeats)||repeats<1||repeats>1000) throw new Error("반복 횟수는 1~20의 정수입니다.");
  const result=[];
  for(const x of values($("batchX").value)) for(const z of values($("batchZ").value)) for(const yaw of values($("batchYaw").value))
    for(let repeat=0;repeat<repeats;repeat++) result.push({x,z,yaw,repeat});
  if(result.length>1000) throw new Error("한 번에 최대 1,000조건까지 실행합니다.");
  return result;
}
function batchRow(report) {
  const row=document.createElement("tr"),o=report.options;
  [`${fmt(o.x,2)} / ${fmt(o.z,2)} / ${fmt(o.yaw,1)}° · 시드 ${o.seed}`,report.state,reasonText(report),`${fmt(report.elapsed,2)} / ${fmt(report.compute_seconds,2)} s`,`${report.collision?"충돌":"없음"} / 잔여 ${fmt(report.true_remaining)} m`].forEach(text=>{const cell=document.createElement("td");cell.textContent=text;row.append(cell);});
  const cell=document.createElement("td"),button=document.createElement("button");button.textContent="재현";
  button.onclick=()=>exclusive(async()=>{
    await releaseSession();
    $("connectionStatus").textContent="선택한 조건을 다시 계산하고 있습니다.";
    const rerun=await api("run",{options:report.options,overrides:report.overrides,capture:true,show_window:true});
    setOptions(report.options);setOverrides(report.overrides);await loadReport(rerun,true);
    $("connectionStatus").textContent=rerun.trace_hash===report.trace_hash?"동일 조건 재현 완료 · 전체 입력/상태/명령 해시 일치":"재현 해시 불일치 · 코드 또는 실행 환경을 확인하세요.";
  });cell.append(button);row.append(cell);$("batchResults").append(row);
}
function updateBatchSummary(total) {
  const rs=app.batches,passed=rs.filter(r=>r.success).length,done=rs.filter(r=>r.state==="DONE").length;
  const reasons={};rs.forEach(r=>r.reasons.forEach(reason=>{reasons[reason]=(reasons[reason]||0)+1;}));
  $("batchSummary").textContent=`${rs.length}/${total} 완료 · 삽입 검증 통과 ${passed} · FSM DONE ${done}\n`+Object.entries(reasons).map(([r,n])=>`${r}: ${n}`).join(" / ");
  controls();
}
async function runBatch() {
  const poses=conditions(),base=options(),config=overrides();
  const explicit=$("batchSeeds").value.trim();
  const seeds=explicit?values(explicit):null;
  if(seeds&&seeds.some(s=>!Number.isSafeInteger(s)||s<0)) throw new Error("시드는 0 이상의 정수로 입력하세요.");
  const unique=poses.filter(c=>c.repeat===0);
  const cases=seeds?unique.flatMap(({repeat,...pose})=>seeds.map(seed=>({options:{...base,...pose,placement_mode:'relative',seed},overrides:config}))):
    poses.map(({repeat,...pose})=>({options:{...base,...pose,placement_mode:'relative',seed:base.seed+repeat},overrides:config}));
  if(cases.length>1000) throw new Error("최대 1,000건까지 실행할 수 있어요.");
  await releaseSession();
  app.batches=[];app.cancel=false;$("batchResults").replaceChildren();$("cancelBatchBtn").disabled=false;
  try {
    const job=await api('batch/start',{cases,workers:Number($("batchWorkers").value)});
    let cursor=0,cancelSent=false;
    while(true) {
      if(app.cancel&&!cancelSent) {await api('batch/cancel',{id:job.id});cancelSent=true;}
      const state=await api('batch/status',{id:job.id,cursor});cursor=state.cursor;
      for(const report of state.results) {app.batches.push(report);batchRow(report);}
      updateBatchSummary(state.total);
      $("batchSummary").textContent+=`\n${state.workers}개 병렬 · ${({running:"진행 중",completed:"완료",cancelled:"중단",error:"오류"})[state.status]||state.status} · ${state.seconds.toFixed(1)}초`;
      $("batchGroups").textContent=state.groups.map(g=>`X ${g.x} / Z ${g.z} / 각도 ${g.yaw}: ${g.passed}/${g.completed} 성공 (${(100*g.passed/g.completed).toFixed(1)}%)`).join('\n');
      if(state.status!=='running') {if(state.error) throw new Error(state.error);break;}
      await new Promise(resolve=>setTimeout(resolve,500));
    }
  } finally {$("cancelBatchBtn").disabled=true;}
}
async function gridSearch() {
  await releaseSession();
  let sets=[overrides()];
  for(const line of $("gridSpec").value.split("\n").filter(s=>s.trim())) {
    const parts=line.split("=");
    if(parts.length!==2||!app.info.grid_fields.includes(parts[0].trim())) throw new Error("지원되는 설정 이름=숫자,숫자 형식으로 입력하세요.");
    sets=sets.flatMap(set=>values(parts[1]).map(value=>({...set,[parts[0].trim()]:value})));
    if(sets.length>50) throw new Error("설정 조합은 최대 50개입니다.");
  }
  const cases=conditions(),base=options();
  if(sets.length*cases.length>1000) throw new Error("총 실행 수는 1,000개 이하여야 합니다.");
  app.grids=[];app.cancel=false;$("gridResults").replaceChildren();$("cancelBatchBtn").disabled=false;
  try {
    let count=0;
    for(const config of sets) {
      const reports=[];
      for(const c of cases) {
        if(app.cancel) break;
        const {repeat,...pose}=c;
        reports.push(await api("run",{options:{...base,...pose,placement_mode:"relative",seed:base.seed+repeat},overrides:config}));
        $("gridStatus").textContent=`${++count}/${sets.length*cases.length} 계산 완료`;
      }
      if(reports.length!==cases.length) break;
      app.grids.push({overrides:config,reports,passed:reports.filter(r=>r.success).length});
    }
    app.grids.sort((a,b)=>b.passed-a.passed);
    for(const result of app.grids) {
      const button=document.createElement("button");button.textContent=`${result.passed}/${cases.length} 통과 · ${JSON.stringify(result.overrides)} · 적용`;
      button.onclick=()=>exclusive(async()=>{setOverrides(result.overrides);await reset();});$("gridResults").append(button);
    }
    $("gridStatus").textContent+=app.cancel?" · 중단 (완료한 설정만 표시)":" · 탐색 완료";
  } finally {$("cancelBatchBtn").disabled=true;}
}

function drawRecording(frame) {
  if(!app.recording) return;
  const ctx=app.recording.canvas.getContext("2d");
  ctx.fillStyle="white";ctx.fillRect(0,0,1280,800);
  // Keep the square coordinate plane undistorted in the exported video.
  ctx.drawImage($("simCanvas"),280,60,720,720);
  ctx.fillStyle="#20242a";ctx.font="16px sans-serif";
  ctx.fillText(`WORLD X-Z | t = ${fmt(frame.t,3)} s`,20,30);

}
function startVideo() {
  if(typeof MediaRecorder==="undefined"||!HTMLCanvasElement.prototype.captureStream) throw new Error("이 브라우저는 화면 녹화를 지원하지 않습니다.");
  const canvas=document.createElement("canvas");canvas.width=1280;canvas.height=800;
  const stream=canvas.captureStream(30),chunks=[];
  const recorder=new MediaRecorder(stream);
  recorder.ondataavailable=e=>{if(e.data.size) chunks.push(e.data);};
  recorder.onstop=()=>{app.video=new Blob(chunks,{type:recorder.mimeType});stream.getTracks().forEach(t=>t.stop());app.recording=null;$("downloadVideoBtn").disabled=false;$("startVideoBtn").disabled=false;$("videoStatus").textContent="녹화 완료";};
  app.recording={canvas,recorder};recorder.start();
  if(app.displayFrame) drawRecording(app.displayFrame);
  $("startVideoBtn").disabled=true;$("stopVideoBtn").disabled=false;$("videoStatus").textContent="녹화 중";
}

async function animate(now) {
  requestAnimationFrame(animate);
  if(app.serverClock&&!app.report&&!app.externalWorld) {
    if(!app.running||app.busy||now-app.lastWall<100) return;
    app.busy=true;app.lastWall=now;
    try {
      const started=performance.now(),simBefore=app.displayFrame?.t??0;
      const result=await api('poll',{id:app.session,cursor:app.serverCursor,display_samples:app.displaySamples||[]});
      app.displaySamples=[];
      const pollMs=performance.now()-started;
      app.serverCursor=result.cursor;
      if(result.frames.length) {
        app.lastRenderTiming=null;
        await appendFrames(result.frames);
        const finished=performance.now(),last=app.displayFrame;
        if(last) {
          app.displaySamples.push({browser_mono_ms:finished,display_wall_utc:new Date().toISOString(),
            clock_domain:'server_display',sim_time_s:last.t,state:last.state,command:last.command,
            sim_advance_s:last.t-simBefore,cycle_ms:finished-started,poll_request_ms:pollMs,
            poll_lock_wait_ms:result.server_lock_wait_ms,
            interval_ms:app.lastPerfWall==null?null:finished-app.lastPerfWall,
            document_hidden:document.hidden,time_scale:Number($("timeScale").value),
            frame_count:app.frames.length,...app.lastRenderTiming});
          app.lastPerfWall=finished;
        }
      }
      if(result.done) {
        app.running=false;showReport(result.report);
        for(const sample of app.displaySamples) await api('performance',{id:app.session,sample});
        app.displaySamples=[];
      }
    } catch(error) {app.running=false;$("connectionStatus").textContent=error.message;}
    finally {app.busy=false;controls();}
    return;
  }
  if(app.externalWorld&&!app.busy&&now-app.lastWall>200) {
    app.lastWall=now;app.busy=true;
    try {
      const d=await api("world/diagnostics",{id:app.externalWorld});
      app.activeOptions=d.options;
      const frame={t:d.sim_time_s,state:"EXTERNAL FSM",command:`CAN ${127-d.can.data[2]}/${127-d.can.data[1]}`,
        truth:d.truth,observation:d.observation,model_packet:d.model_packet,collision:d.collision,clearance:d.clearance,
        can:d.can,bus_status:d.can.status,can_frame_count:d.can.received_frames,
        stable_frames:0,target:null,lines:[]};
      if(!app.frames.length||app.frames.at(-1).t!==frame.t) await appendFrames([frame]);
      else await render(frame);
    } catch(e) {$("externalStatus").textContent=e.message;app.externalWorld=null;}
    finally {app.busy=false;controls();}
    return;
  }
  if(app.externalWorld) return;
  if(!app.running) {app.lastWall=now;return;}
  // Preserve elapsed wall time while a step/render/ACK is in flight.
  if(app.busy) return;
  const rate=Number($("timeScale").value),hz=app.activeOptions?.hz||Number($("hz").value);
  const count=Math.min(120,Math.floor((now-app.lastWall)*rate*hz/1000));
  if(count<1) return;
  const duration=Math.min(1,(now-app.lastWall)*rate/1000);
  const before=app.displayFrame?.t ?? 0;
  app.busy=true;
  try {
    await step(count,duration);
    // Charge only actual simulation progress. The server can return less
    // than requested at its tick cap; retain the rest for the next batch.
    if(app.running) app.lastWall+=Math.max(0,(app.displayFrame?.t ?? before)-before)*1000/rate;
  } catch(error) {app.running=false;$("connectionStatus").textContent=error.message;}
  finally {app.busy=false;controls();}
}
function bind() {
  const panel=document.querySelector('.control-panel');
  const sections=[...panel.children];
  const columns=[0,1].map(()=>{const column=document.createElement('div');column.className='control-column';panel.append(column);return column;});
  sections.forEach((section,index)=>{section.style.order=index;columns[index%2].append(section);});
  placementControls();
  $("placement_mode").addEventListener("change",placementControls);
  const updateView=()=>{
    const zoom=Number($("viewZoom").value),x=Number($("viewCenterX").value),z=Number($("viewCenterZ").value);
    if(!Number.isFinite(zoom)||zoom<25||zoom>400||!Number.isFinite(x)||!Number.isFinite(z)) return;
    app.view={zoom,x,z};
    if(app.displayFrame) {draw(app.displayFrame);drawRecording(app.displayFrame);}
  };
  ["viewZoom","viewCenterX","viewCenterZ"].forEach(id=>$(id).addEventListener("input",updateView));
  $("resetZoomBtn").onclick=()=>{$("viewZoom").value="100";updateView();};
  $("runBtn").onclick=toggleRun;
  $("stepBtn").onclick=()=>exclusive(()=>step());
  ["resetBtn","applyScenarioBtn","applyConstantsBtn"].forEach(id=>$(id).onclick=()=>exclusive(reset));
  $("clearConstantsBtn").onclick=()=>exclusive(async()=>{setOverrides({});await reset();});
  $("fastBtn").onclick=()=>exclusive(async()=>{ await releaseSession();$("connectionStatus").textContent="계산 중";await loadReport(await api("run",{options:options(),overrides:overrides(),capture:true,show_window:true}),true);$("connectionStatus").textContent="계산 완료";});
  $("runBatchBtn").onclick=()=>exclusive(runBatch);$("runGridBtn").onclick=()=>exclusive(gridSearch);
  $("cancelBatchBtn").onclick=()=>{app.cancel=true;$("cancelBatchBtn").disabled=true;};
  $("replaySlider").oninput=()=>exclusive(async()=>{app.cursor=Number($("replaySlider").value);await render(app.frames[app.cursor]);});
  $("timeScale").onchange=async()=>{
    if(app.serverClock&&app.running&&!app.report) {
      try {await api('play',{id:app.session,rate:Number($("timeScale").value)});}
      catch(error) {$("connectionStatus").textContent=error.message;}
    }
  };
  $("downloadTrajectoryJsonBtn").onclick=()=>exclusive(async()=>download(JSON.stringify(await fullReport(),null,2),"application/json","fsm_v4_run.json"));
  $("downloadTrajectoryCsvBtn").onclick=()=>exclusive(async()=>{
    const r=await fullReport();download(csv([["t_s","state","command","truth_x_m","truth_z_m","truth_yaw_deg","world_x_m","world_z_m","detected","collision","model_sequence","model_fps","model_x_m","model_y_m","model_z_m","model_yaw_deg","error_x_m","error_y_m","error_z_m","error_yaw_deg"],...r.frames.map(f=>[f.t,f.state,f.command,f.truth.x,f.truth.z,f.truth.yaw,f.truth.world_x,f.truth.world_z,f.observation.detected,!!f.collision,f.observation.sequence,f.observation.result_fps,f.observation.x,f.observation.y,f.observation.z,f.observation.yaw,...["x","y","z","yaw"].map(k=>f.observation.sampled_error?.[k])])]),"text/csv","fsm_v4_trajectory.csv");
  });
  $("downloadBatchBtn").onclick=()=>download(csv([["x_m","z_m","yaw_deg","seed","perception","success","fsm_state","failure_state","reason","seconds","true_remaining_m","collision","source_id","trace_hash","options_json","overrides_json"],...app.batches.map(r=>[r.options.x,r.options.z,r.options.yaw,r.options.seed,r.options.perception,r.success,r.state,r.failure_state,r.reasons.join(" / "),r.elapsed,r.true_remaining,!!r.collision,r.source_id,r.trace_hash,JSON.stringify(r.options),JSON.stringify(r.overrides)])]),"text/csv","fsm_v4_conditions.csv");
  $("downloadBatchJsonBtn").onclick=()=>download(JSON.stringify({provenance:app.info,reports:app.batches,grids:app.grids},null,2),"application/json","fsm_v4_experiments.json");
  $("replayCheckBtn").onclick=()=>exclusive(async()=>{
    $("replayCheckStatus").textContent="동일 관측을 원본 FSM에 다시 입력하고 있습니다.";
    const r=await fullReport();if(r.replayable===false) throw new Error("프로세스 장애 기록은 정상 실행 재현 검사 대상이 아닙니다.");
    const result=await api("replay-check",{trace:r.trace,overrides:r.overrides,source_id:r.source_id,simulator_id:r.simulator_id});
    $("replayCheckStatus").textContent=`${result.frames}프레임 비교 · ${result.passed?"모두 일치":"불일치 "+result.mismatches.length+"개"}`;
  });
  $("importRun").onchange=()=>exclusive(async()=>{
    const file=$("importRun").files[0];if(!file)return;
    const r=JSON.parse(await file.text());
    if(!Array.isArray(r.frames)||!r.frames.length||!Array.isArray(r.trace)||!r.options||!r.config||!Array.isArray(r.reasons)) throw new Error("실행 JSON 형식이 아닙니다.");
    if(r.frames.some(f=>!f.truth||!f.observation||!Array.isArray(f.can?.data)||!Array.isArray(f.lines))) throw new Error("프레임 데이터가 잘못되었습니다.");
    await releaseSession();
    setOptions(r.options);setOverrides(r.overrides||{});await loadReport(r);
    $("connectionStatus").textContent=r.source_id===app.info.source_id&&r.simulator_id===app.info.simulator_id?"저장된 실행을 불러왔습니다.":"다른 FSM 또는 시뮬레이터 버전의 기록입니다. 화면 재생만 가능하며 원본 일치 검증은 차단됩니다.";
  });
  $("startVideoBtn").onclick=()=>{try{startVideo();}catch(e){$("videoStatus").textContent=e.message;}};
  $("stopVideoBtn").onclick=()=>{app.recording?.recorder.stop();$("stopVideoBtn").disabled=true;};
  $("downloadVideoBtn").onclick=()=>download(app.video,app.video.type,app.video.type.includes("mp4")?"fsm_v4.mp4":"fsm_v4.webm");
  $("attachWorldBtn").onclick=()=>exclusive(async()=>{
    const id=$("externalWorldId").value.trim();
    const d=await api("world/diagnostics",{id});
    await releaseSession();
    app.externalWorld=id;app.session=null;app.report=null;app.frames=[];app.cursor=0;app.lastWall=0;app.activeOptions=d.options;
    $("externalStatus").textContent="외부 FSM의 CAN 제어를 관찰 중 · 이 화면에서는 차량을 조작하지 않습니다.";
    $("resultReadout").textContent="외부 FSM 연결 · 상태 이름 대신 수신 CAN과 차량 상태를 표시합니다.";
  });
  $("detachWorldBtn").onclick=()=>exclusive(async()=>{app.externalWorld=null;$("externalStatus").textContent="내장 v4 어댑터 사용 중";await reset();});
  $("downloadCanBtn").onclick=()=>exclusive(async()=>{
    const frames=app.externalWorld?(await api("world/can-log",{id:app.externalWorld})).frames:(await fullReport()).can_frames;
    download(JSON.stringify({protocol:"lift.can.v1",frames},null,2),"application/json","virtual_can_frames.json");
  });
  document.querySelectorAll("[data-option]").forEach(e=>e.addEventListener("change",()=>{$("connectionStatus").textContent="입력값이 변경되었습니다. 배치 적용 또는 끝까지 계산 시 반영됩니다.";}));
  window.addEventListener("resize",()=>{if(app.displayFrame){draw(app.displayFrame);}});
  window.addEventListener("pagehide",()=>{
    if(app.session) navigator.sendBeacon('/api/session/close',new Blob([JSON.stringify({id:app.session})],{type:'application/json'}));
  });
}
async function releaseSession() {
  LiftCameraView.reset();
  if(app.session) {await api("session/close",{id:app.session});app.session=null;}
}
function applyInfo(info) {
  app.info=info;
  $("sourceStatus").textContent=`${info.controller}\nFSM 별도 프로세스 · 소스 ${info.source_id.slice(0,16)} · 새 실행마다 저장된 코드 로드`;
  $("sourceDetails").textContent=JSON.stringify(info,null,2);
  document.querySelectorAll("[data-config]").forEach(input=>{input.placeholder=info.config[input.dataset.config];});
}
async function init() {
  try {
    const presetText=new URLSearchParams(location.search).get('preset');
    const preset=presetText ? JSON.parse(presetText) : null;
    if(preset) setOptions(preset.options || {});
    // This is a placement preview, not an FSM result. It needs no server/model.
    const initialOptions=options();
    app.displayFrame={t:0,truth:LiftViewGeometry.initialTruth(initialOptions),collision:null};
    draw(app.displayFrame);
    $("connectionStatus").textContent="초기 배치 미리보기 · 서버 연결 중";
    bind();
    applyInfo(await api("info"));
    app.info.grid_fields.forEach(key=>{const label=document.createElement("label");label.textContent=key;const input=document.createElement("input");input.type="number";input.step="any";input.dataset.config=key;input.placeholder=app.info.config[key];label.append(input);$("constantGrid").append(label);});
    if(preset) setOverrides(preset.overrides || {});
    await reset();controls();requestAnimationFrame(animate);
  } catch(error) {$("connectionStatus").textContent=`초기 배치 미리보기 · 연결 실패: ${error.message} · 기존 서버를 종료하고 run_v4_simulator.bat를 다시 실행하세요.`;}
}
init();
