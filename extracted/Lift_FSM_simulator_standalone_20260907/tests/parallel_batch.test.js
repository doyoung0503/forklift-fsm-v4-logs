const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
test('explicit seeds form a Cartesian batch with one start request and incremental results',async()=>{
 const nodes={batchX:{value:'0,.3'},batchZ:{value:'4'},batchYaw:{value:'15'},batchRepeats:{value:'1'},batchSeeds:{value:'11,12'},batchWorkers:{value:'2'},batchResults:{replaceChildren(){}},cancelBatchBtn:{},batchSummary:{textContent:''},batchGroups:{}};
 const calls=[];
 const ctx=vm.createContext({Image:class{},LiftViewGeometry:{palletImage:{src:''}},document:{getElementById:id=>nodes[id]},calls});
 const source=fs.readFileSync(path.join(__dirname,'../v4_sim.js'),'utf8').replace(/init\(\);\s*$/,'');
 vm.runInContext(source+`
 options=()=>({seed:10});overrides=()=>({});releaseSession=async()=>{};batchRow=()=>{};updateBatchSummary=()=>{};
 api=async(op,data)=>{calls.push({op,data});return op==='batch/start'?{id:'j'}:{cursor:1,status:'completed',total:4,workers:2,seconds:1,directory:'results',groups:[],results:[{success:true}]};};
 this.run=runBatch;`,ctx);
 await ctx.run();
 assert.equal(calls[0].op,'batch/start');
 assert.equal(calls[0].data.cases.length,4);
 assert.deepEqual(Array.from(calls[0].data.cases,c=>c.options.seed),[11,12,11,12]);
 assert.equal(calls[0].data.workers,2);
 assert.equal(calls[1].op,'batch/status');
 assert.equal(nodes.cancelBatchBtn.disabled,true);
});
