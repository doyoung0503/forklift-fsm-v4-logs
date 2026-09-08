const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');

test('coordinate plane, clock and recording wait for the native displayed frame',async()=>{
  const nodes=new Map(),jobs=[],painted=[];
  const ctx=vm.createContext({
    document:{getElementById:id=>{if(!nodes.has(id))nodes.set(id,{dataset:{}});return nodes.get(id);}},
    Image:class{},LiftViewGeometry:{palletImage:{src:''}},
    LiftCameraView:{update:(frame,options,commit)=>new Promise(resolve=>jobs.push({frame,commit,resolve}))},
    painted,
  });
  let source=fs.readFileSync(path.join(__dirname,'../v4_sim.js'),'utf8').replace(/init\(\);\s*$/,'');
  vm.runInContext(source+`
    draw=f=>painted.push(['world',f.t]);
    drawRecording=f=>painted.push(['recording',f.t]);
    app.activeOptions={};app.session='native';
    this.ui={app,render,step,controls,toggleRun};
  `,ctx);
  const frame=t=>({t,state:'STATE_'+t,command:'STOP',truth:{},observation:{},can:{data:[127]},lines:[]});
  const old=frame(1),next=frame(2);
  ctx.ui.app.frames=[old,next];
  let first=ctx.ui.render(old);jobs[0].commit(old,{window_open:true});jobs[0].resolve(true);await first;
  painted.length=0;
  ctx.ui.app.report={};ctx.ui.app.cursor=0;
  let finished=false;
  const stepping=ctx.ui.step().then(()=>finished=true);
  await Promise.resolve();
  assert.equal(finished,false);
  assert.equal(nodes.get('simClock').textContent,'t = 1.000 s');
  assert.equal(nodes.get('simCanvas').dataset.simTime,'1');
  assert.deepEqual(painted,[]);
  ctx.ui.app.info={};ctx.ui.app.running=true;ctx.ui.app.busy=true;
  ctx.ui.controls();assert.equal(nodes.get('runBtn').disabled,false);
  ctx.ui.toggleRun();assert.equal(ctx.ui.app.running,false);
  assert.equal(finished,false);  // Pause does not start another tick or skip the pending display.
  jobs[1].commit(next,{window_open:true});jobs[1].resolve(true);await stepping;
  assert.equal(finished,true);
  assert.equal(nodes.get('simClock').textContent,'t = 2.000 s');
  assert.equal(nodes.get('simCanvas').dataset.simTime,'2');
  assert.deepEqual(JSON.parse(JSON.stringify(painted)),[['world',2],['recording',2]]);
});
