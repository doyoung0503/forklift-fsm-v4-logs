"use strict";
// Render once, present that immutable frame in the native window, then commit
// all browser panels. Presentation never advances virtual time or the FSM.
const LiftCameraView = (() => {
  let pending=null, busy=false, generation=0, serial=0;
  function reset() {
    generation++;
    if(pending) pending.resolve(false);
    pending=null;
  }
  async function post(route,data) {
    const response=await fetch(`/api/${route}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result=await response.json();
    if(!response.ok) throw new Error(result.error||`${route}: HTTP ${response.status}`);
    return result;
  }
  async function pump() {
    if(busy||!pending) return;
    busy=true;const job=pending;pending=null;const epoch=generation;
    const canvas=document.getElementById('cameraCanvas');
    const status=document.getElementById('cameraStatus');
    let committed=false;
    try {
      let result=null,picture=null,jpeg=null,cameraError=null;
      try {
        result=await post('camera',{frame:job.frame,options:job.options});
        if(result.t!==job.frame.t) throw new Error('카메라 프레임 시각 불일치');
        if(typeof result.jpeg!=='string'||!result.jpeg) throw new Error('카메라 이미지 누락');
        jpeg=result.jpeg;
        // The coordinate-only browser forwards the image to the FSM window.
        // A camera canvas is optional for clients that still show a preview.
        if(canvas) {
          picture=new Image();picture.src='data:image/jpeg;base64,'+jpeg;
          await picture.decode();
        }
      } catch(error) {cameraError=error;picture=null;jpeg=null;}
      if(epoch!==generation||pending) return;
      let ack=null;
      if(job.session) {
        ack=await post('present',{id:job.session,presentation_id:job.id,frame:job.frame,
          options:job.options,jpeg});
        if(ack.error) throw new Error(ack.error);
        if(ack.presentation_id!==job.id||ack.t!==job.frame.t)
          throw new Error('FSM 표시 프레임 확인 불일치');
      }
      if(epoch!==generation) return;
      // Commit an accepted frame even if a newer seek arrived during the ACK.
      if(canvas&&picture) canvas.getContext('2d').drawImage(picture,0,0,640,480);
      else if(canvas) {
        const ctx=canvas.getContext('2d');ctx.fillStyle='#18202a';ctx.fillRect(0,0,640,480);
        ctx.fillStyle='#fff';ctx.font='18px sans-serif';ctx.fillText('카메라 렌더링 오류',20,40);
      }
      if(canvas) canvas.dataset.simTime=String(job.frame.t);
      const sync=ack?(ack.window_open?'FSM 창 동기화':'FSM 창 닫힘'):'브라우저 재생';
      status.textContent=`t=${job.frame.t.toFixed(3)}s · ${sync}${cameraError?' · '+cameraError.message:''}`;
      job.commit(job.frame,ack);committed=true;
    } catch(error) {
      if(epoch===generation) {
        status.textContent=`화면 동기화 오류: ${error.message}`;
        job.onError(error);
      }
    } finally {
      job.resolve(committed);busy=false;if(pending) pump();
    }
  }
  function update(frame,options,commit,session=null,onError=()=>{}) {
    return new Promise(resolve=>{
      if(pending) pending.resolve(false);
      pending={frame,options,commit,session,onError,resolve,id:++serial};pump();
    });
  }
  return {update,reset};
})();
