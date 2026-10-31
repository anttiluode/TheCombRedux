"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const E=require("../engine.js");
function close(a,b,tol=1e-10){assert.ok(Math.abs(a-b)<tol,"expected "+a+" ~ "+b);}
function run(){
 const check=E.verify();
 assert.equal(check.replay,true);
 const model=E.train(52026,180,50);
 assert.equal(model.weights.length,4);
 assert.equal(model.weights[0].length,17);
 // A comb tooth is truly fitted from synthetic examples, not a hard-coded XOR branch.
 assert.ok(E.opPredict(model.operators[0],1)>.85,"copy tooth should learn bit 1");
 assert.ok(E.opPredict(model.operators[1],1)<.15,"invert tooth should flip bit 1");
 assert.ok(E.opPredict(model.operators[0],0)<.15);
 assert.ok(E.opPredict(model.operators[1],0)>.85);
 // An isolated XOR observation cannot distinguish (0,0) from (1,1).
 const ambiguous=new E.Bayes();ambiguous.observe({x:0,y:0,toolProbe:null,worldProbe:null});
 close(ambiguous.p[0],ambiguous.p[3],1e-9);
 // A tool calibration breaks part of the ambiguity; calibrated world probe resolves both.
 let p=ambiguous.observe({x:0,y:0,toolProbe:0,worldProbe:0});
 assert.ok(p[0]>.95,"independent measurements must resolve cause");
 // Exact-seed replay includes identical hidden truth, probes, beliefs and predictions.
 let a=new E.Lab(model,772),b=new E.Lab(model,772);
 for(let k=0;k<65;k++){
  let opt={hazards:true,probe:k%11===0?"tool":k%19===0?"world":"none"};
  assert.deepEqual(a.step(opt),b.step(opt));
 }
 // The recurrent vector is a real dependency of the workspace and the comb.
 const testLab=new E.Lab(model,99);
 for(let k=0;k<15;k++)testLab.step({probe:k%5===0?"tool":"none",x:k%2});
 const oldVec=testLab.ring.v.slice();
 assert.equal(oldVec.length,16);
 assert.ok(oldVec.slice(2).some(x=>Math.abs(x)>1e-4));
 const pLive=testLab.workspace();
 testLab.ablations.cutRead=true;
 const pCut=testLab.workspace();
 assert.ok(pCut.some((v,i)=>Math.abs(v-pLive[i])>1e-5),"cutting read changes decoder");
 const kept=testLab.ring.v.slice();
 assert.deepEqual(kept,oldVec,"read cut must not erase stored state");
 testLab.ablations.cutRead=false;testLab.ablations.mask14=true;
 const pMask=testLab.workspace();
 assert.ok(pMask.some((v,i)=>Math.abs(v-pLive[i])>1e-5),"high coordinates must actually be used");
 // Bayes is isolated: workspace ablations do not touch its posterior.
 const bayesA=new E.Lab(model,12),bayesB=new E.Lab(model,12);
 bayesB.ablations.erase=true;
 for(let t=0;t<40;t++){
   const pa=bayesA.step({hazards:true,probe:t%8===0?"tool":"none"});
   const pb=bayesB.step({hazards:true,probe:t%8===0?"tool":"none"});
   assert.deepEqual(pa.bayes,pb.bayes);
   assert.deepEqual(pa.truth,pb.truth);
 }
 const result=E.evaluate(model,8301,160);
 const ablated=E.evaluate(model,8301,160,{erase:true});
 assert.ok(Number.isFinite(result.brier));
 assert.equal(result.probes,ablated.probes);
 assert.ok(Math.abs(result.brier-ablated.brier)>1e-6,"ablation must change scored model");
 const index=fs.readFileSync(path.join(__dirname,"..","index.html"),"utf8");
 assert.ok(index.includes('src="engine.js"'));
 const inline=index.match(/<script>([\s\S]*?)<\/script>/);
 assert.ok(inline,"inline browser application exists");
 new vm.Script(inline[1],{filename:"index.html:inline"});
 assert.ok(index.includes("mask14")&&index.includes("cutRead"));
 console.log(JSON.stringify({status:"PASS",checks:check,intact:result,erased:ablated},null,2));
}
run();
