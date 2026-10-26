"use strict";
const el=id=>document.getElementById(id), val=(o,...keys)=>keys.reduce((a,k)=>a&&typeof a==="object"?a[k]:undefined,o);
const num=x=>typeof x==="number"&&Number.isFinite(x)?x:null;
const fmt=x=>num(x)===null?"—":new Intl.NumberFormat("en-US",{maximumFractionDigits:0}).format(x);
const usd=x=>num(x)===null?"—":new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",maximumFractionDigits:2}).format(x);
const txt=(id,value)=>{el(id).textContent=value===null||value===undefined?"—":String(value);};
const node=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=String(text);return n;};
const roles=[['scanner','SCANNER','#d4f477'],['narrative','NARRATIVE','#83d8e9'],['risk','RISK','#ff8484'],['perps','PERPS','#b9b4f1'],['world','WORLD','#ead37b'],['verifier','VERIFIER','#9fe8b5']];
let snapshot=null,product="spot",chart="equity",rawVisible=false;
let paused=localStorage.getItem("fieldlab-motion-paused")==="true";
function setMotion(){document.body.classList.toggle("motion-paused",paused);el("motion").setAttribute("aria-pressed",String(paused));txt("motion",paused?"Resume motion":"Pause motion");}
setMotion();el("motion").addEventListener("click",()=>{paused=!paused;localStorage.setItem("fieldlab-motion-paused",String(paused));setMotion();});
el("raw-toggle").addEventListener("click",()=>{rawVisible=!rawVisible;el("raw-panel").hidden=!rawVisible;el("raw-toggle").setAttribute("aria-expanded",String(rawVisible));if(rawVisible&&snapshot)txt("raw-data",JSON.stringify(snapshot,null,2));});
for(const [index,[role,label,color]] of roles.entries()){
  const wrap=node('div',`robot-wrap ${role}`);wrap.id=`bot-${role}`;wrap.style.setProperty('--x',`${7+index*15}%`);wrap.style.setProperty('--speed',`${4+index*.7}s`);wrap.style.setProperty('--bot',color);
  wrap.append(node('div','robot-label',label));const bubble=node('div','bubble','No event observed yet');bubble.id=`bubble-${role}`;wrap.append(bubble);
  const bot=node('div','robot'),head=node('div','robot-head');head.append(node('i','eye'),node('i','eye'));bot.append(head,node('div','robot-body'),node('i','arm left'),node('i','arm right'),node('i','leg left'),node('i','leg right'));wrap.append(node('div','shadow'),bot);el('room-stage').append(wrap);
}
document.querySelectorAll('[data-product]').forEach(button=>button.addEventListener('click',()=>{product=button.dataset.product;document.querySelectorAll('[data-product]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-selected',String(b===button));});render();}));
document.querySelectorAll('[data-chart]').forEach(button=>button.addEventListener('click',()=>{chart=button.dataset.chart;document.querySelectorAll('[data-chart]').forEach(b=>b.classList.toggle('selected',b===button));render();}));
function facts(id,rows){const box=el(id);box.replaceChildren();for(const [label,value,tone] of rows){const line=node('div');line.append(node('dt','',label),node('dd',tone||'',value??'—'));box.append(line);}}
function gauge(label,a,b,detail){const p=num(a)!==null&&num(b)!==null&&b>0?Math.max(0,Math.min(100,100*a/b)):null;const row=node('div','gauge-row'),ring=node('div','ring');ring.style.setProperty('--progress',String(p??0));ring.append(node('span','',p===null?'—':`${p.toFixed(p>99&&p<100?2:p<10?1:0)}%`));const text=node('div','gauge-label',label);text.append(node('small','',detail));row.append(ring,text);return row;}
function sourceFresh(frame){return !!frame?.available&&!frame.last_error_type&&num(frame.cache_age_seconds)!==null&&frame.cache_age_seconds<25;}
function render(){
  if(!snapshot)return; document.querySelector('.chart-tabs').hidden=product!=='spot';
  const spot=snapshot.sources?.spot||{},world=snapshot.sources?.world||{},main=spot.data||{},w=world.data||{};
  const portfolio=main.portfolio||{},fork=main.native_fork_probe||{},nlp=main.neural_sentiment||{},perps=main.phantom_perps||{},canonical=val(main,'unknown_opportunity_engine','canonical_evidence')||{};
  const locked=spot.health?.live_submission_locked===true&&world.health?.live_submission_locked===true;
  const fresh=sourceFresh(spot),bothFresh=fresh&&sourceFresh(world);txt('mode',bothFresh&&locked?'PAPER / LIVE LOCKED':bothFresh?'VERIFY SOURCE MODE':'TELEMETRY DELAYED');el('mode').classList.toggle('warning',!bothFresh||!locked);
  const age=num(spot.cache_age_seconds);txt('freshness',age===null?'Awaiting source snapshot':`Monitor cache ${Math.floor(age)}s · upstream age shown below`);
  const notes={spot:'Solana launch research · unknown costs, sellability and unproven edge remain visible.',world:'Phantom World only · no Kalshi or other venue substitution. Quotes, rules and eligibility require verification.',perps:'Public underlying-perps research · catalog membership does not prove availability in Phantom. Independent controls are not strategy alpha.'};txt('product-note',notes[product]);
  const byVenue=fork.by_venue||{},venueRows=Object.values(byVenue),untried=venueRows.reduce((a,r)=>a+(num(r.untried_targets)??0),0);
  el('gauges').replaceChildren(gauge('RETAINED TARGETS TRIED',num(fork.targets)===null?null:fork.targets-untried,fork.targets,`${fmt(fork.targets)} queued · ${fmt(untried)} untried`),gauge('NLP INPUTS COMPLETE',nlp.completed_inputs,nlp.inputs,`${fmt(nlp.pending_inputs)} inputs pending`),gauge('COMPUTED WINDOWS CANONICAL',nlp.canonical_windows,nlp.retained_windows,`${fmt(nlp.awaiting_canonical_ack)} awaiting ACK`));
  if(product==='spot'){
    txt('hero-title',chart==='equity'?'PAPER MODEL EQUITY':'RETAINED EVIDENCE TAPE');txt('hero-number',chart==='equity'?usd(portfolio.portfolio_value):fmt(canonical.records));txt('hero-unit',chart==='equity'?'USD · reported model capital, not wallet balance':'canonical records · not unique launches');
    txt('decision',portfolio.kill_switch_active?'MODEL FROZEN':'PAPER / EVIDENCE FIRST');txt('decision-detail',main.promotion_blocked_at?.reason||'Software validation does not establish a positive trading edge.');
    txt('hero-delta',`reported model daily P&L ${usd(portfolio.daily_pnl)}`);
    txt('position-state',`${fmt(portfolio.open_positions)} OPEN`);txt('position-head',num(portfolio.open_positions)===0?'FLAT / OBSERVING':'MODEL POSITIONS');txt('hold-label',locked?'LIVE SUBMISSION LOCKED':'LOCK STATE UNVERIFIED');
    facts('position-facts',[['Open model positions',fmt(portfolio.open_positions)],['Exposure · USD',usd(portfolio.exposure_usd)],['Reported model equity',usd(portfolio.portfolio_value)],['Kill switch',portfolio.kill_switch_active===true?'ACTIVE':portfolio.kill_switch_active===false?'not active':'—',portfolio.kill_switch_active?'bad':'']]);
    txt('position-caption','Read-only model account. A sampled local-bank success is not a live fill or a guarantee of a future exit.');
  }else if(product==='world'){
    txt('hero-title','PHANTOM WORLD / VERIFICATION');txt('hero-number','UNVERIFIED');txt('hero-unit','No verified executable World quote is inferred from controller health.');txt('decision','VENUE DATA REQUIRED');txt('decision-detail','Rules, eligibility, quote freshness and settlement identity remain explicit requirements.');txt('hero-delta','No invented prediction fills');
    txt('position-state',world.health?.mode||'NO CONTROLLER SNAPSHOT');txt('position-head','WORLD RESEARCH');txt('hold-label','NO VENUE SUBSTITUTION');
    facts('position-facts',[['Controller',world.health?.status||'unavailable'],['Health mode',world.health?.mode||'—'],['Executable World quotes','not verified','warn'],['Paper venue P&L','not established','warn']]);txt('position-caption','Other-venue prices or historical control fills are not relabeled as Phantom World trading results. Inspect raw data for the controller report.');
  }else{
    txt('hero-title','PUBLIC UNDERLYING PERPS');txt('hero-number',fmt(perps.active_markets));txt('hero-unit',`${fmt(perps.observed_catalog_dexes)} observed catalogs · Phantom eligibility is separate`);txt('decision','RESEARCH / CONTROLS');txt('decision-detail','Observed books and funding are data assets; static scenarios and control fills do not prove a strategy edge.');txt('hero-delta',`${fmt(perps.funding_records)} funding observations`);
    txt('position-state',perps.status||'UNAVAILABLE');txt('position-head','CATALOG / BOOKS');txt('hold-label','SEPARATE RESEARCH CAPITAL');
    facts('position-facts',[['Active underlying markets',fmt(perps.active_markets)],['Books recorded',fmt(perps.books_recorded)],['Funding rows',fmt(perps.funding_records)],['Control fills',fmt(perps.simulated_control_fills)]]);txt('position-caption','Underlying market observations do not certify that every contract is tradable through Phantom. No orders can be placed from this terminal.');
  }
  const list=el('venue-list');list.replaceChildren();
  for(const [name,row] of Object.entries(byVenue)){
    const box=node('div','venue-row'),line=node('div','venue-name');line.append(node('span','',name.replaceAll('_',' ')),node('span','',`${fmt(row.attempts)} attempts`));
    const track=node('div','track'),bar=node('i');const progress=num(row.targets)&&num(row.untried_targets)!==null?100*(row.targets-row.untried_targets)/row.targets:0;bar.style.width=`${Math.max(0,Math.min(100,progress))}%`;track.append(bar);
    box.append(line,track,node('small','',`${fmt(row.targets)} retained · ${fmt(row.untried_targets)} untried · ${row.coverage_state||'unknown'}`));list.append(box);
  }
  if(!list.childElementCount)list.append(node('div','empty','No per-venue snapshot yet.'));
  facts('risk-facts',[['Main live lock',spot.health?.live_submission_locked===true?'LOCKED':'UNVERIFIED',spot.health?.live_submission_locked===true?'good':'warn'],['World live lock',world.health?.live_submission_locked===true?'LOCKED':'UNVERIFIED',world.health?.live_submission_locked===true?'good':'warn'],['Ledger write failures',fmt(canonical.write_failures),canonical.write_failures?'bad':''],['Recovery mode',canonical.recovery_mode||'—'],['World quote eligibility','unverified','warn']]);
  txt('risk-footer','No automatic promotion. Positive after-cost edge and whole-market coverage are not established here.');
  facts('intelligence-facts',[['Canonical NLP windows',fmt(nlp.canonical_windows)],['NLP backlog',fmt(nlp.pending_inputs),'warn'],['Worker calls',fmt(nlp.inference_metrics?.worker_inferences)],['Exact-window cache hits',fmt(nlp.inference_metrics?.cache_hits)],['CPU / inter-op threads',`${fmt(nlp.inference_profile?.intra_op_threads)} / ${fmt(nlp.inference_profile?.inter_op_threads)}`],['Source-linked targets',fmt(fork.canonical_native_linked_targets)],['Watched entities',fmt(main.authenticity?.watched_entities)]]);txt('nlp-state',nlp.trading_skill||'UNMEASURED');
  const sourceList=el('source-list');sourceList.replaceChildren();
  for(const [name,frame] of Object.entries(snapshot.sources||{})){
    const item=node('div','source-item'),good=sourceFresh(frame),title=node('strong','',name==='spot'?'MEMECOIN DESK':'PHANTOM WORLD CONTROLLER'),status=node('span',`source-state ${good?'':'delayed'}`,good?(frame.health?.status||'snapshot'):'DELAYED / UNAVAILABLE');
    item.append(title,status);const upstreamAt=num(frame.data?.diagnostic_snapshot?.observed_at);const upstreamAge=upstreamAt===null?'unreported':`${Math.max(0,Math.floor(snapshot.observed_at-upstreamAt))}s`;
    item.append(node('p','',`Cache ${num(frame.cache_age_seconds)===null?'—':Math.floor(frame.cache_age_seconds)+'s'} · upstream snapshot ${upstreamAge} · ${frame.last_error_type||'no transport error reported'}`));sourceList.append(item);
  }
  const events=snapshot.events||[];renderEvents(events);renderBots(events,fresh);drawGrid(fork);drawChart();
  if(rawVisible)txt('raw-data',JSON.stringify(snapshot,null,2));
  txt('footer-data',`${fmt(canonical.records)} canonical records · ${fmt(snapshot.event_count_since_start)} monitor events since start`);
}
function renderEvents(events){const feed=el('event-feed');feed.replaceChildren();if(!events.length){feed.append(node('div','empty','Waiting for real service changes. No synthetic conversation is generated.'));return;}
  for(const event of events.slice(-70).reverse()){
    const row=node('div','event'),head=node('div','event-header');row.style.setProperty('--event-color',roles.find(r=>r[0]===event.role)?.[2]||'#d4f477');head.append(node('strong','',String(event.role).toUpperCase()),node('span','',event.source),node('time','',new Date(event.observed_at*1000).toLocaleTimeString('en-GB')));row.append(head,node('p','',event.text));feed.append(row);
  }
}
function renderBots(events,online){for(const [role] of roles){const wrap=el(`bot-${role}`),latest=events.findLast(e=>e.role===role);wrap.classList.toggle('idle',!online);const recent=latest&&snapshot.observed_at-latest.observed_at<18;wrap.classList.toggle('speaking',Boolean(recent));txt(`bubble-${role}`,latest?.text||'No event observed yet.');}}
function context(canvas){const rect=canvas.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.max(1,Math.floor(rect.width*ratio));canvas.height=Math.max(1,Math.floor(rect.height*ratio));const c=canvas.getContext('2d');c.scale(ratio,ratio);return [c,rect.width,rect.height];}
function drawChart(){const [c,w,h]=context(el('equity-chart'));c.clearRect(0,0,w,h);c.strokeStyle='#40533b55';c.lineWidth=1;for(let y=20;y<h;y+=32){c.beginPath();c.setLineDash([2,5]);c.moveTo(0,y);c.lineTo(w,y);c.stroke();}c.setLineDash([]);
  const key=product==='perps'?'perps_funding_records':chart==='equity'?'model_equity_usd':'canonical_records';const points=(snapshot?.history||[]).filter(p=>num(p[key])!==null);const empty=product==='world'||points.length<2;
  el('chart-empty').hidden=!empty;txt('chart-empty',product==='world'?'World price / fill history unavailable · no venue substitution':'Waiting for actual status observations');
  txt('chart-scope',product==='perps'?'Observed funding-record counter · not price or P&L':product==='world'?'Verified World-only data required':chart==='equity'?'Reported model equity snapshots · not confirmed net fill history':'Retained evidence counter · not a launch census');
  if(empty)return;const values=points.map(p=>p[key]),low=Math.min(...values),high=Math.max(...values),pad=Math.max((high-low)*.15,Math.abs(high)*.00001,1),min=low-pad,max=high+pad;
  const t0=points[0].observed_at,t1=points.at(-1).observed_at;const xy=points.map(p=>[2+(w-4)*(p.observed_at-t0)/Math.max(1,t1-t0),h-10-(h-25)*(p[key]-min)/(max-min)]);const color=values.at(-1)<values[0]?'#ff8484':'#d4f477';
  c.beginPath();xy.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.lineTo(w,h);c.lineTo(0,h);c.closePath();const grad=c.createLinearGradient(0,15,0,h);grad.addColorStop(0,color+'36');grad.addColorStop(1,color+'02');c.fillStyle=grad;c.fill();
  c.beginPath();xy.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.strokeStyle=color;c.lineWidth=2;c.stroke();const last=xy.at(-1);c.beginPath();c.arc(last[0],last[1],4,0,Math.PI*2);c.fillStyle=color;c.shadowColor=color;c.shadowBlur=12;c.fill();c.shadowBlur=0;
}
function drawGrid(fork){const [c,w,h]=context(el('scan-grid')),d=fork.canonical_dispositions||{},good=num(d.ROUNDTRIP_SUCCEEDED_IN_FORK)??0,total=Object.values(d).reduce((a,n)=>a+(num(n)??0),0),failed=(num(d.FORK_BUY_FAILED_UNKNOWN)??0)+(num(d.FORK_SELL_FAILED)??0),unknown=Math.max(0,total-good-failed);txt('scan-total',`${fmt(total)} RECORDED`);c.clearRect(0,0,w,h);
  const columns=15,rows=9,gap=3,cw=(w-gap*(columns-1))/columns,ch=(h-gap*(rows-1))/rows,count=columns*rows;
  for(let i=0;i<count;i++){const share=total?i/count:null;c.fillStyle=share===null?'#293b30':share<good/total?'#b6db73':share<(good+unknown)/total?'#796f40':'#b46f6b';c.globalAlpha=.65+(i%3)*.12;c.fillRect((i%columns)*(cw+gap),Math.floor(i/columns)*(ch+gap),cw,ch);}c.globalAlpha=1;
}
window.addEventListener('resize',()=>{if(snapshot){drawChart();drawGrid(snapshot.sources?.spot?.data?.native_fork_probe||{});}});
setInterval(()=>txt('clock',new Date().toLocaleTimeString('en-GB')+' LOCAL'),1000);
async function poll(){try{const response=await fetch('/api/snapshot',{cache:'no-store'});if(!response.ok)throw new Error('monitor unavailable');snapshot=await response.json();render();}catch{txt('freshness','Monitor connection unavailable');txt('mode','CONNECTION LOST');el('mode').classList.add('warning');}finally{setTimeout(poll,3000);}}
poll();
