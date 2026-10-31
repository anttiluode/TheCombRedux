"use strict";
const E=require("./engine.js");
const seeds=[6102,6103,6104,6111,6112,6113,6220,6221,6222,6223,6224,6225];
const arms=[["intact",{}],["cut_read",{cutRead:true}],["erase_each_step",{erase:true}],["mask_coordinates_2_to_15",{mask14:true}]];
function mean(xs){return xs.reduce((a,b)=>a+b,0)/xs.length;}
function std(xs){let m=mean(xs);return Math.sqrt(xs.reduce((s,x)=>s+(x-m)*(x-m),0)/(xs.length-1));}
function run(){
 const model=E.train(52026,340,50);
 const conditions=arms.map(([name,ablations])=>{
 const rows=seeds.map(seed=>E.evaluate(model,seed,320,ablations));
 return {name,accuracy:mean(rows.map(x=>x.accuracy)),accuracy_sd:std(rows.map(x=>x.accuracy)),brier:mean(rows.map(x=>x.brier)),brier_sd:std(rows.map(x=>x.brier)),per_seed:rows.map(x=>({seed:x.seed,accuracy:x.accuracy,brier:x.brier}))};
 });
 const reference=E.evaluate(model,seeds[0],320,{});
 const baseline=seeds.map(seed=>E.evaluate(model,seed,320));
 const result={description:"Exploratory matched-stream experiment; frozen trained readout; exact same diagnostic schedule; state affects decisions but is not guaranteed to be useful",
 training:{seed:52026,episodes:340,steps:50},
 evaluation:{seeds,n_per_seed:320,probe_schedule:"tool every 17 steps, world every 31 steps except tool priority"},
 reference:{bayesian_accuracy:mean(baseline.map(x=>x.oracleAccuracy)),static_accuracy:mean(baseline.map(x=>x.staticAccuracy))},
 conditions,
 matched_effects:{
  intact_minus_erase_accuracy:mean(conditions[0].per_seed.map((x,i)=>x.accuracy-conditions[2].per_seed[i].accuracy)),
  intact_minus_erase_brier:mean(conditions[0].per_seed.map((x,i)=>x.brier-conditions[2].per_seed[i].brier)),
  intact_minus_cut_accuracy:mean(conditions[0].per_seed.map((x,i)=>x.accuracy-conditions[1].per_seed[i].accuracy))
 }};
 console.log(JSON.stringify(result,null,2));
}
run();
