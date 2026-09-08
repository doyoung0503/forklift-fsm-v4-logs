const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');

function setup(withCanvas=true) {
  const requests=[],commits=[],errors=[],images=[];
  const canvas={dataset:{},getContext:()=>({drawImage(){},fillRect(){},fillText(){}})};
  const status={textContent:''};
  const context=vm.createContext({document:{getElementById:id=>id==='cameraCanvas'?(withCanvas?canvas:null):status},
    Image:class {constructor(){images.push(this);} async decode(){}},fetch:(url,init)=>new Promise(resolve=>requests.push({url,data:JSON.parse(init.body),resolve}))});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../camera_view.js'),'utf8')+'\nthis.api=LiftCameraView;',context);
  const update=(t,session=null)=>context.api.update({t},{},f=>commits.push(f.t),session,e=>errors.push(e));
  const finish=async(index,extra={})=>{
    requests[index].resolve({ok:true,json:async()=>({t:requests[index].data.frame.t,jpeg:'jpeg',model_sequence:5,render_ms:4,
      presentation_id:requests[index].data.presentation_id,window_open:true,...extra})});
    await new Promise(resolve=>setImmediate(resolve));
  };
  return {context,requests,commits,canvas,update,finish,errors,images};
}
test('coalesces requests and commits the matching 2D frame with each image',async()=>{
  const s=setup();s.update(1);s.update(2);s.update(3);
  assert.equal(s.requests.length,1);await s.finish(0);
  assert.deepEqual(s.commits,[]);assert.equal(s.requests[1].data.frame.t,3);
  await s.finish(1);assert.deepEqual(s.commits,[3]);assert.equal(s.canvas.dataset.simTime,'3');
});

test('browser waits for native ACK of the same frame and exact camera JPEG',async()=>{
  const s=setup();const done=s.update(4,'session');
  await s.finish(0);assert.deepEqual(s.commits,[]);
  assert.equal(s.requests[1].url,'/api/present');
  assert.equal(s.requests[1].data.jpeg,'jpeg');
  assert.equal(s.requests[1].data.frame.t,4);
  await s.finish(1);assert.equal(await done,true);
  assert.deepEqual(s.commits,[4]);assert.equal(s.canvas.dataset.simTime,'4');
});

test('coordinate-only browser forwards JPEG to native window without a camera canvas',async()=>{
  const s=setup(false);const done=s.update(4,'session');
  await s.finish(0);assert.deepEqual(s.commits,[]);
  assert.equal(s.requests[1].data.jpeg,'jpeg');assert.equal(s.images.length,0);
  await s.finish(1);assert.equal(await done,true);
  assert.deepEqual(s.commits,[4]);assert.deepEqual(s.errors,[]);
});

test('stale native ACK never commits a mismatched browser frame',async()=>{
  const s=setup();const done=s.update(4,'session');
  await s.finish(0);await s.finish(1,{presentation_id:-1});
  assert.equal(await done,false);assert.deepEqual(s.commits,[]);assert.equal(s.errors.length,1);
});

test('backwards replay commits accepted frame before the pending seek',async()=>{
  const s=setup();const first=s.update(8,'session');await s.finish(0);
  const next=s.update(2,'session');await s.finish(1);
  assert.equal(await first,true);assert.deepEqual(s.commits,[8]);
  await s.finish(2);await s.finish(3);
  assert.equal(await next,true);assert.deepEqual(s.commits,[8,2]);
});

test('reset resolves discarded pending jobs and discards an old native ACK',async()=>{
  const s=setup();const first=s.update(8,'old');await s.finish(0);
  const pending=s.update(9,'old');s.context.api.reset();
  assert.equal(await pending,false);await s.finish(1);
  assert.equal(await first,false);assert.deepEqual(s.commits,[]);
});

test('camera failure presents an empty camera for this time, not the previous image',async()=>{
  const s=setup();const done=s.update(4,'session');
  s.requests[0].resolve({ok:false,json:async()=>({error:'render failed'})});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(s.requests[1].data.jpeg,null);await s.finish(1);
  assert.equal(await done,true);assert.equal(s.canvas.dataset.simTime,'4');
});

test('native rendering failure preserves the previous complete browser frame',async()=>{
  const s=setup();const first=s.update(1,'session');await s.finish(0);await s.finish(1);await first;
  const next=s.update(2,'session');await s.finish(2);await s.finish(3,{error:'native draw failed'});
  assert.equal(await next,false);assert.deepEqual(s.commits,[1]);
  assert.equal(s.canvas.dataset.simTime,'1');assert.equal(s.errors.length,1);
});
test('reset discards an in-flight old-session frame',async()=>{
  const s=setup();s.update(10);s.context.api.reset();s.update(0);
  await s.finish(0);assert.deepEqual(s.commits,[]);
  await s.finish(1);assert.deepEqual(s.commits,[0]);
});
