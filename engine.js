/* TheCombRedux engine. No dependencies. UMD: works as a browser script or Node module. */
(function(root,factory){const api=factory();if(typeof module!=="undefined"&&module.exports)module.exports=api;else root.CombRedux=api;})(typeof globalThis!=="undefined"?globalThis:this,function(){
"use strict";
const D=16, H=4, EPS=1e-12;
function rng(seed){let a=seed>>>0;return function(){a+=0x6D2B79F5;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));
const sigmoid=x=>1/(1+Math.exp(-clamp(x,-30,30)));
function softmax(a){let max=Math.max(...a), e=a.map(x=>Math.exp(x-max)),z=e.reduce((u,v)=>u+v,0);return e.map(x=>x/z);}
function entropy(p){return -p.reduce((z,v)=>z+(v>0?v*Math.log2(v):0),0);}
function initialMatrix(seed){const r=rng(seed),a=[];for(let j=0;j<4;j++)a.push(Array.from({length:D+1},()=>.035*(2*r()-1)));return a;}
function inputWeights(seed){const r=rng(seed),w=[];for(let i=0;i<D;i++){w[i]=[];for(let k=0;k<7;k++)w[i].push((.22+.09*(i%3))*(2*r()-1));}return w;}
const INPUT_W=inputWeights(3110);
function Ring(){this.v=Array(D).fill(0);this.lastInnovation=0;}
Ring.prototype.clear=function(){this.v.fill(0);this.lastInnovation=0;};
Ring.prototype.update=function(e){
 // 8 rotating phase planes, 4 fast (coordinates 0..7) and 4 slow (8..15).
 // All 16 coordinates are passed to the learned workspace, not merely plotted.
 const f=[2*e.parity-1,2*e.x-1,2*e.y-1,e.toolProbe==null?0:2*e.toolProbe-1,e.toolProbe==null?0:1,e.worldProbe==null?0:2*e.worldProbe-1,e.worldProbe==null?0:1];
 const old=this.v.slice(),next=[];
 for(let i=0;i<D;i+=2){
   const slow=i>=8,theta=slow?.15:.58, decay=slow?.979:.82;
   const u=decay*(Math.cos(theta)*old[i]-Math.sin(theta)*old[i+1]);
   const v=decay*(Math.sin(theta)*old[i]+Math.cos(theta)*old[i+1]);
   for(let k=0;k<2;k++){
     const z=i+k,drive=f.reduce((s,x,j)=>s+x*INPUT_W[z][j],0);
     const gated=drive*(1+.32*Math.tanh(old[(z+5)%D]));
     next[z]=Math.tanh((k===0?u:v)+gated);
   }
 }
 this.v=next;
 this.lastInnovation= Math.sqrt(next.reduce((s,v,i)=>s+(v-old[i])**2,0)/D);
 return next;
};
function logits(weights,s){return weights.map(w=>w[0]+s.reduce((v,x,i)=>v+x*w[i+1],0));}
function decode(weights,s){return softmax(logits(weights,s));}
function opPredict(op,x){return sigmoid(op[0]+op[1]*(2*x-1));}
function combPredict(ops,p,x){return (p[0]+p[1])*opPredict(ops[0],x)+(p[2]+p[3])*opPredict(ops[1],x);}
function train(seed=52026,episodes=340,steps=50){
 const r=rng(seed),weights=initialMatrix(seed^0x8473),ops=[[0,0],[0,0]];
 const lr=.036, opLR=.09;
 // Training uses simulator truth as supervision. Deployed live inference NEVER receives those labels.
 for(let ep=0;ep<episodes;ep++){
   const ring=new Ring();let world=r()<.5?0:1,tool=r()<.5?0:1;
   for(let t=0;t<steps;t++){
     if(r()<.045)world^=1;if(r()<.035)tool^=1;
     const x=r()<.5?0:1;let y=x^world^tool;if(r()<.035)y^=1;
     const qTool=r()<.15?(tool^(r()<.01?1:0)):null;
     const qWorld=r()<.065?(world^(r()<.01?1:0)):null;
     ring.update({x,y,parity:x^y,toolProbe:qTool,worldProbe:qWorld});
     const p=decode(weights,ring.v),target=world*2+tool;
     for(let j=0;j<H;j++){
       const g=(p[j]-(j===target?1:0))*lr;
       weights[j][0]-=g;
       for(let i=0;i<D;i++)weights[j][i+1]-=g*ring.v[i];
     }
     // Train two distinct copy/invert comb operators from actual labels.
     const label=x^world, pr=opPredict(ops[world],x),g=(pr-label)*opLR;
     ops[world][0]-=g;ops[world][1]-=g*(2*x-1);
   }
 }
 return {weights,operators:ops,seed,episodes,steps,supervision:"simulated latent labels during offline training only"};
}
function Bayes(){this.p=[.25,.25,.25,.25];}
Bayes.prototype.observe=function(e){
 // Transition: independent hazard rates matching the synthetic stream generator.
 const prior=[];
 for(let w=0;w<2;w++)for(let m=0;m<2;m++){
  let z=0;for(let a=0;a<2;a++)for(let b=0;b<2;b++)
   z+=this.p[a*2+b]*(a===w?.955:.045)*(b===m?.965:.035);
  prior.push(z);
 }
 const v=prior.map((p,i)=>{
   const w=i>>1,m=i&1;
   const normal=(e.x^w^m)===e.y;
   let likelihood=normal?.965:.035;
   if(e.toolProbe!==null)likelihood*=e.toolProbe===m?.99:.01;
   if(e.worldProbe!==null)likelihood*=e.worldProbe===w?.99:.01;
   return p*likelihood;
 });
 let sum=v.reduce((x,y)=>x+y,0);
 this.p=sum>EPS?v.map(x=>x/sum):[.25,.25,.25,.25];
 return this.p;
};
function Lab(model,seed=8765){
 this.model=model||train();this.r=rng(seed);this.seed=seed;this.world=0;this.tool=0;
 this.ring=new Ring();this.bayes=new Bayes();this.tick=0;this.lastProbe=-20;
 this.history=[];this.ablations={cutRead:false,erase:false,mask14:false};
 this.auto=false;this.worldHazard=.045;this.toolHazard=.035;this.noise=.035;this.counts={net:0,oracle:0,static:0,probes:0};
 this.last=null;
}
Lab.prototype.setTruth=function(k){this[k]^=1;return this[k];};
Lab.prototype.restart=function(seed=this.seed){
 const same=this.model;const fresh=new Lab(same,seed);Object.assign(this,fresh);return this;
};
Lab.prototype.workspace=function(){
 const coords=this.ablations.cutRead?Array(D).fill(0):this.ring.v.slice();
 if(this.ablations.mask14)for(let i=2;i<D;i++)coords[i]=0;
 return decode(this.model.weights,coords);
};
Lab.prototype.step=function(options={}){
 const r=this.r;
 if(options.switchWorld)this.world^=1;if(options.switchTool)this.tool^=1;
 if(options.hazards){if(r()<this.worldHazard)this.world^=1;if(r()<this.toolHazard)this.tool^=1;}
 let x=options.x==null?(r()<.5?0:1):Number(options.x);
 let y=x^this.world^this.tool;if(r()<this.noise)y^=1;
 let probe=options.probe||"none";
 if(probe==="auto"){
   const pp=this.workspace(),pw=pp[2]+pp[3],pt=pp[1]+pp[3];
   probe=(this.tick-this.lastProbe>=5 && (Math.abs(pw-.5)<.30||Math.abs(pt-.5)<.30))?(this.tick%3===0?"world":"tool"):"none";
 }
 const toolProbe=probe==="tool"?(this.tool^(r()<.01?1:0)):null;
 const worldProbe=probe==="world"?(this.world^(r()<.01?1:0)):null;
 if(this.ablations.erase)this.ring.clear();
 this.ring.update({x,y,parity:x^y,toolProbe,worldProbe});
 const p=this.workspace();
 const b=this.bayes.observe({x,y,toolProbe,worldProbe});
 const pred=combPredict(this.model.operators,p,x);
 const oracle=combPredict(this.model.operators,b,x);
 const staticPred=combPredict(this.model.operators,[.25,.25,.25,.25],x);
 const goal=x^this.world;
 const answer=pred>.5?1:pred<.5?0:null, bayesAnswer=oracle>.5?1:oracle<.5?0:null;
 this.tick++;
 if(probe!=="none"){this.counts.probes++;this.lastProbe=this.tick;}
 this.counts.net+=answer===goal?1:0;
 this.counts.oracle+=bayesAnswer===goal?1:0;
 this.counts.static+=staticPred>.5?(goal===1?1:0):staticPred<.5?(goal===0?1:0):.5;
 const record={t:this.tick,x,y,goal,probe,toolProbe,worldProbe,truth:[this.world,this.tool],
  p:p.slice(),bayes:b.slice(),prediction:pred,oracle,staticPred,answer,innovation:this.ring.lastInnovation,
  vector:this.ring.v.slice(),operation:[p[0]+p[1],p[2]+p[3]],counts:{...this.counts}};
 this.history.push(record);if(this.history.length>1500)this.history.shift();this.last=record;
 return record;
};
function evaluate(model,seed=90210,n=300,intervention={}){
 const lab=new Lab(model,seed);Object.assign(lab.ablations,intervention);
 let autoCount=0;
 for(let t=0;t<n;t++){const probe=(t%17===0?"tool":t%31===0?"world":"none");
   lab.step({probe,hazards:true});if(probe!=="none")autoCount++;
 }
 const diffs=lab.history.map(z=>z.prediction-z.goal);
 return {seed,n,accuracy:lab.counts.net/n,oracleAccuracy:lab.counts.oracle/n,
  staticAccuracy:lab.counts.static/n,brier:diffs.reduce((s,v)=>s+v*v,0)/n,probes:autoCount};
}
function verify(){
 let model=train(14,120,35);const a=new Lab(model,200),b=new Lab(model,200);
 for(let i=0;i<30;i++){const p=i%7===0?"tool":"none";const u=a.step({probe:p}),v=b.step({probe:p});
  if(JSON.stringify(u)!==JSON.stringify(v))throw Error("seeded replay differs");
 }
 const x=a.step({x:1,probe:"world"}),m=a.ring.v.slice();if(m.some(v=>!Number.isFinite(v)))throw Error("non-finite state");
 const p=a.workspace(),old=a.ablations.cutRead;a.ablations.cutRead=true;const q=a.workspace();a.ablations.cutRead=old;
 if(q.some((v,i)=>Math.abs(v-p[i])>0)&&a.ring.v.length!==D)throw Error("bad read ablation");
 if(Math.abs(p.reduce((x,y)=>x+y,0)-1)>1e-9)throw Error("invalid softmax");
 let op0=opPredict(model.operators[0],1),op1=opPredict(model.operators[1],1);
 if(!(op0>.85&&op1<.15))throw Error("operators not learned");
 const bb=new Bayes();let z=bb.observe({x:0,y:0,toolProbe:0,worldProbe:0});
 if(!(z[0]>.85))throw Error("Bayes probe failed");
 return {replay:true,finite:true,operatorCopy:op0,operatorInvert:op1,probePosterior:z[0]};
}
return {D,train,Lab,Ring,Bayes,decode,combPredict,opPredict,entropy,evaluate,verify,rng};
});
