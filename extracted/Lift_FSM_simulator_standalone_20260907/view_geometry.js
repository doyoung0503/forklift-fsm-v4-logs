(function(root) {
  "use strict";
  const geometry = {
    // Original PNG, with only transparent outer padding excluded at draw time.
    palletImage: {src:"assets/pallet_top.png", crop:[81,75,1887,1898]},
    initialTruth(options,config={}) {
      const heading=options.forklift_heading??0;
      let x=options.x??0,z=options.z??2,yaw=options.yaw??0;
      if(options.placement_mode==='world') {
        const a=heading*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
        const dx=options.pallet_x-options.forklift_x,dz=options.pallet_z-options.forklift_z;
        x=c*dx-s*dz+(config.CAMERA_TO_ROT_CENTER_X_M??0);
        z=s*dx+c*dz+(config.CAMERA_TO_ROT_CENTER_Z_M??-.68);
        yaw=((options.pallet_heading-180-heading)%360+540)%360-180;
      }
      return {x,z,yaw,heading,world_x:options.forklift_x??0,world_z:options.forklift_z??0};
    },
    dimensions(config,options) {
      const cameraToTip=config.CAMERA_TO_FORK_TIP_Z_M??1.18;
      const pivotZ=config.CAMERA_TO_ROT_CENTER_Z_M??-.68;
      return {width:config.PALLET_FRONT_VISIBILITY_WIDTH_M??1.1,depth:options.pallet_depth??1.1,
        opening:config.INSERT_OPENING_SPAN_M??.71,forkSpan:config.FORK_OUTER_SPAN_M??.6,
        cameraToTip,pivotZ,pivotToTip:cameraToTip-pivotZ};
    },
    palletPoint(truth,x,z) {
      const a=truth.yaw*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
      return [truth.x+c*x+s*z,truth.z-s*x+c*z];
    },
    // Image top = pallet back; image bottom = the selected insertion face.
    imageTransform(project,width,depth) {
      const origin=project(-width/2,depth),right=project(width/2,depth),bottom=project(-width/2,0);
      return [right[0]-origin[0],right[1]-origin[1],bottom[0]-origin[0],bottom[1]-origin[1],...origin];
    },
    viewport(w,h,zoom=100,cx=0,cz=0) {
      const span=1000/zoom,side=Math.max(1,Math.min(w,h));
      const left=(w-side)/2,top=(h-side)/2;
      return {span,side,left,top,scale:side/span,
        xmin:cx-span/2,xmax:cx+span/2,zmin:cz-span/2,zmax:cz+span/2};
    },
    fovPolygon(view,camera,heading,hfov) {
      // Clip the viewport against two infinite half-planes, without a far cap.
      let points=[[view.xmin,view.zmin],[view.xmax,view.zmin],[view.xmax,view.zmax],[view.xmin,view.zmax]];
      const a=heading*Math.PI/180,c=Math.cos(a),s=Math.sin(a),t=Math.tan(hfov*Math.PI/360);
      for(const sign of [-1,1]) {
        const distance=([x,z])=>{const dx=x-camera[0],dz=z-camera[1];return t*(s*dx+c*dz)+sign*(c*dx-s*dz);};
        const clipped=[];
        for(let i=0;i<points.length;i++) {
          const p=points[i],q=points[(i+1)%points.length],dp=distance(p),dq=distance(q);
          if(dp>=0) clipped.push(p);
          if((dp>=0)!==(dq>=0)) {
            const k=dp/(dp-dq);clipped.push([p[0]+k*(q[0]-p[0]),p[1]+k*(q[1]-p[1])]);
          }
        }
        points=clipped;
      }
      return points;
    },
    worldPoint(truth,config,x,z) {
      const a=truth.heading*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
      const dx=x-(config.CAMERA_TO_ROT_CENTER_X_M??0),dz=z-(config.CAMERA_TO_ROT_CENTER_Z_M??-.68);
      return [truth.world_x+c*dx+s*dz,truth.world_z-s*dx+c*dz];
    }
  };
  if(typeof module!=="undefined"&&module.exports) module.exports=geometry;
  else root.LiftViewGeometry=geometry;
})(typeof window!=="undefined"?window:this);
