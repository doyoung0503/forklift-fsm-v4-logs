const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
test('busy time and capped progress remain pending; paused time is discarded',async()=>{
  let finish;const durations=[];
  const ctx=vm.createContext({Image:class{},LiftViewGeometry:{palletImage:{src:''}},
    requestAnimationFrame:()=>{},document:{getElementById:()=>({value:'1'})},durations,
    startStep:(duration)=>new Promise(r=>{finish=r;})});
  const src=fs.readFileSync(path.join(__dirname,'../v4_sim.js'),'utf8').replace(/init\(\);\s*$/,'');
  vm.runInContext(src+`
    controls=()=>{};step=async(count,duration)=>{durations.push(duration);await startStep(duration);app.displayFrame.t+=.4;};
    app.running=true;app.lastWall=0;app.activeOptions={hz:30};app.displayFrame={t:0};
    this.ui={app,animate};`,ctx);
  const a=ctx.ui.animate(1000);
  await ctx.ui.animate(1300);assert.equal(ctx.ui.app.lastWall,0);
  finish();await a;assert.equal(ctx.ui.app.lastWall,400);
  const b=ctx.ui.animate(1500);assert.equal(durations[1],1);
  finish();await b;assert.equal(ctx.ui.app.lastWall,800);
  ctx.ui.app.running=false;await ctx.ui.animate(9000);assert.equal(ctx.ui.app.lastWall,9000);
});

test('server display samples piggyback on polls and final display is flushed',async()=>{
  const calls=[];let now=100;
  const ctx=vm.createContext({Image:class{},LiftViewGeometry:{palletImage:{src:''}},
    requestAnimationFrame:()=>{},document:{hidden:false,getElementById:()=>({value:'1'})},
    performance:{now:()=>++now},calls});
  const src=fs.readFileSync(path.join(__dirname,'../v4_sim.js'),'utf8').replace(/init\(\);\s*$/,'');
  vm.runInContext(src+`
    controls=()=>{};showReport=r=>{app.report=r;};
    api=async(op,data)=>{calls.push({op,data});return op==='poll'?{cursor:calls.length,frames:[{t:calls.length,state:'TEST'}],done:calls.length===2,report:{}}:{};};
    appendFrames=async(frames)=>{app.frames.push(...frames);app.displayFrame=frames.at(-1);app.lastRenderTiming={browser_paint_ms:2};};
    app.serverClock=true;app.running=true;app.session='test';app.serverCursor=0;app.lastWall=0;
    this.ui={app,animate};`,ctx);
  await ctx.ui.animate(200);
  assert.equal(calls.length,1);
  assert.equal(ctx.ui.app.displaySamples[0].interval_ms,null);
  await ctx.ui.animate(400);
  assert.equal(calls[1].data.display_samples.length,1);
  assert.equal(calls[2].op,'performance');
  assert.ok(calls[2].data.sample.interval_ms>0);
  assert.equal(calls[2].data.sample.browser_paint_ms,2);
  assert.equal(ctx.ui.app.displaySamples.length,0);
});
