const assert=require('node:assert/strict');
const {viewport,worldPoint,dimensions,palletPoint,imageTransform,fovPolygon}=require('../view_geometry.js');
for(const [w,h] of [[400,360],[1200,500],[320,800]]) {
  const v=viewport(w,h,100);
  assert.equal(v.xmax-v.xmin,10);
  assert.equal(v.zmax-v.zmin,10);
  assert.equal(v.scale*10,v.side);
  assert.equal(viewport(w,h,200).span,5);
  assert.equal(viewport(w,h,50).span,20);
  const moved=viewport(w,h,100,3,-4);
  assert.equal(moved.scale,v.scale);
  assert.equal((moved.xmin+moved.xmax)/2,3);
}
const config={CAMERA_TO_ROT_CENTER_X_M:0,CAMERA_TO_ROT_CENTER_Z_M:-.68};
const p=worldPoint({world_x:1,world_z:2,heading:90},config,0,2);
assert.ok(Math.abs(p[0]-3.68)<1e-10);
assert.ok(Math.abs(p[1]-2)<1e-10);
console.log('PASS fixed 10m square, zoom, viewport centre and vehicle world transform');
const profile=require('../vehicle_profile.json').parameters;
const d=dimensions(profile,{pallet_depth:1.1});
assert.equal(d.width,1.1);assert.equal(d.depth,1.1);
assert.equal(d.opening,.71);assert.equal(d.forkSpan,.6);
assert.ok(Math.abs(d.pivotToTip-1.86)<1e-12);
// The visible PNG boundary occupies 1.1m on both axes, at any rotation/zoom.
for(const heading of [0,37,90,180,-135]) for(const zoom of [50,100,200,400]) {
  const truth={world_x:-1,world_z:2,heading,x:.3,z:2,yaw:25};
  const scale=viewport(900,700,zoom).scale;
  const project=(x,z)=>worldPoint(truth,profile,...palletPoint(truth,x,z)).map(v=>v*scale);
  const [a,b,c,e,tx,ty]=imageTransform(project,d.width,d.depth);
  assert.ok(Math.abs(Math.hypot(a,b)-1.1*scale)<1e-9);
  assert.ok(Math.abs(Math.hypot(c,e)-1.1*scale)<1e-9);
  const front=project(0,0);
  assert.ok(Math.hypot(tx+a/2+c-front[0],ty+b/2+e-front[1])<1e-9);
}
console.log('PASS pallet image metric bounds, front anchor, rotation and measured fork geometry');
const farView=viewport(720,720,25,0,30);
assert.equal(Math.max(...fovPolygon(farView,[0,0],0,55).map(p=>p[1])),50);
assert.equal(fovPolygon(farView,[0,0],180,55).length,0);
const distantCamera=fovPolygon(viewport(720,720),[0,-1000],0,55);
assert.equal(distantCamera.length,4);
for(const heading of [0,30,90,180,-90]) {
  const a=heading*Math.PI/180;
  for(const [x,z] of fovPolygon(viewport(720,720,25),[1,-2],heading,55)) {
    const dx=x-1,dz=z+2,forward=Math.sin(a)*dx+Math.cos(a)*dz,lateral=Math.cos(a)*dx-Math.sin(a)*dz;
    assert.ok(forward>=-1e-9);
    assert.ok(Math.abs(lateral)<=forward*Math.tan(55*Math.PI/360)+1e-9);
    assert.ok(Math.abs(x)<=20+1e-9&&Math.abs(z)<=20+1e-9);
  }
}
console.log('PASS unlimited FOV viewport clipping, rotation, and off-screen camera');
