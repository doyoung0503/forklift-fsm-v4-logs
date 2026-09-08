const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
const geometry=require('../view_geometry.js');

test('initial world placement matches camera offset and heading',()=>{
  const o={placement_mode:'world',forklift_x:0,forklift_z:-.68,forklift_heading:0,
    pallet_x:0,pallet_z:2,pallet_heading:180};
  assert.deepEqual(geometry.initialTruth(o),{x:0,z:2,yaw:0,heading:0,world_x:0,world_z:-.68});
  const rotated=geometry.initialTruth({...o,forklift_heading:90,pallet_x:2.68,pallet_z:-.68});
  assert.ok(Math.abs(rotated.x)<1e-10);
  assert.ok(Math.abs(rotated.z-2)<1e-10);
  assert.equal(rotated.yaw,-90);
});

test('startup draws grid, vehicle and pallet even when old server rejects options',async()=>{
  const nodes=new Map(),calls=[];
  const canvas=new Proxy({}, {get:(_,k)=>(...args)=>{calls.push([k,...args]);},set:()=>true});
  const context=vm.createContext({devicePixelRatio:1,Image:class{},LiftViewGeometry:geometry,
    document:{getElementById(id){
      if(!nodes.has(id))nodes.set(id,{dataset:{},getBoundingClientRect:()=>({width:720,height:720}),getContext:()=>canvas});
      return nodes.get(id);
    }}});
  const source=fs.readFileSync(path.join(__dirname,'../v4_sim.js'),'utf8').replace(/init\(\);\s*$/,'');
  vm.runInContext(source+`
    options=()=>({placement_mode:'world',forklift_x:0,forklift_z:-.68,forklift_heading:0,
      pallet_x:0,pallet_z:2,pallet_heading:180,pallet_depth:1.1,fork_width:.1});
    bind=()=>{};
    api=async()=>{throw new Error("Unknown scenario settings: ['inference_mode']");};
    this.start=init;
  `,context);
  await context.start();
  assert.ok(calls.filter(c=>c[0]==='fillText').length>=22,'coordinate grid labels drawn');
  assert.ok(calls.filter(c=>c[0]==='fill').length>=6,'pallet, body and forks drawn');
  assert.ok(nodes.get('worldPoseCaption').textContent.includes('-0.68'));
  assert.ok(nodes.get('connectionStatus').textContent.includes('기존 서버를 종료'));
});
