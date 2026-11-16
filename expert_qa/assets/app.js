'use strict';
const BASE = new URL('.', location.href).pathname;
const el = id => document.getElementById(id);
let snapshot, sessionId = null, busy = false;
const names = {recorded:'Recorded evidence', inference:'Interpretation', limitation:'Limitation'};
const toolNames = {get_research_context:'Research evidence overview',get_analysis:'Analysis results',get_study_effect:'Study effect and source',get_leave_one_out:'Saved leave-one-out results',get_analysis_selection:'Study selection',get_extraction:'Extraction and review',get_screening:'Screening',get_rob:'Risk of bias',get_source:'Source passage',search_case_evidence:'Evidence search',finish_answer:'Compose cited answer'};
function node(tag, text, cls) { const n = document.createElement(tag); if(text !== undefined) n.textContent = text; if(cls) n.className=cls; return n; }
function selected() { return snapshot.analyses.find(a => a.id === el('analysis').value); }
const statuses={completed:'Completed',blocked:'No pooled result',failed:'Analysis incomplete',not_available:'Analysis pending',partial:'Partial results available'};
function fmt(v){return typeof v==='number'&&Number.isFinite(v)?Number(v.toPrecision(4)).toString():'—';}
function articleTitle(aid){const title=snapshot?.articles?.[aid];if(title&&!/^(candidate|a_\d|article_)/i.test(title))return title;const ids=Object.keys(snapshot?.articles||{});const i=ids.indexOf(aid);return i>=0?'Article '+(i+1):'Current analysis';}
function recordLabel(p){const aid=p.arguments?.article_id||snapshot?.evidence_articles?.[p.arguments?.evidence_id];return aid?articleTitle(aid):'Current analysis';}
function refreshAnalysis(){
 const a=selected(),stat=a?.statistics;el('state').textContent=a?(statuses[a.status]||'Result pending review'):'No analysis available';
 el('statistics').textContent=JSON.stringify(a?{analysis_id:a.id,status:a.status,statistics:stat,warnings:a.warnings}:{},null,2);
 const box=el('result-summary');box.replaceChildren();
 if(stat&&stat.pooled_effect!==null&&stat.pooled_effect!==undefined){
  const effectNames={sensitivity:'Sensitivity',specificity:'Specificity',proportion:'Proportion',prevalence:'Prevalence'};
  box.append(node('p',effectNames[stat.effect_type]||stat.effect_type||'Pooled effect','result-label'),node('div',fmt(stat.pooled_effect),'result-value'),node('p',`95% CI ${fmt(stat.ci_lower)} – ${fmt(stat.ci_upper)}`,'result-ci'));
  const facts=node('div',undefined,'result-facts');facts.append(node('span',`${stat.k??'—'} pooled effects`),node('span',stat.model==='fixed_effect'?'Fixed effect':stat.model==='random_effects'?'Random effects':(stat.model||'Saved model').replace(/_/g,' ')));box.append(facts);
 }else box.append(node('p','No pooled estimate','result-label'),node('p','Ask about the available evidence and why it has not been pooled.','muted'));
 reset();
}
function reset(){ if(busy)return; sessionId=null;el('conversation').replaceChildren();el('progress').textContent=''; }
function setBusy(value){ busy=value; for(const n of document.querySelectorAll('button,select,textarea'))n.disabled=value; }
function render(answer){
 const box=node('article',undefined,'answer');box.append(node('strong',answer.mode==='research'?'Research interpretation · Draft for review':'Results interpretation'));
 if(answer.mode==='research')box.append(node('p','These explanations and next steps are proposals for review. Traceable quotations do not establish a hypothesis; the proposed analyses have not been performed.','muted'));
 for(const c of answer.claims){const part=node('div',undefined,'claim');part.append(node('span',names[c.kind],`badge ${c.kind}`),node('p',c.text));const refs=node('div',undefined,'references');for(const id of c.citation_ids){const p=answer.citations[id];const a=node('a',`${toolNames[p.tool]||'Evidence'} · ${recordLabel(p).slice(0,48)}`,'citation');a.href=BASE+'citation/'+encodeURIComponent(id);a.target='_blank';a.rel='noopener';refs.append(a);}part.append(refs);box.append(part);}
 const foot=node('div',undefined,'answer-footer');foot.append(node('span',answer.log_saved?'Conversation saved':'Conversation could not be saved; download a copy','muted'));
 const download=node('button','Download this answer');download.type='button';download.addEventListener('click',()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(answer,null,2)],{type:'application/json'}));const a=node('a');a.href=u;a.download=`expert_qa_${answer.answer_id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});foot.append(download);box.append(foot);el('conversation').append(box);
}
function makeTrace(){
 const box=node('details',undefined,'trace');box.open=true;
 const title=node('summary','Reviewing relevant evidence…');
 const note=node('p','Retrieved materials are listed here. Open each record to inspect the evidence used in the answer.','muted');
 const list=node('div',undefined,'trace-list');const debug=node('details',undefined,'technical');debug.append(node('summary','Run details'));const debugList=node('div');debug.append(debugList);box.append(title,note,list,debug);el('conversation').append(box);
 const pending=[], models=new Map(), pages=[];
 let pageCount=0;
 function event(e){
  const when=e.elapsed_seconds===undefined?'Time not recorded':`${e.elapsed_seconds}s`;
  if(e.type==='model_started'){
   const row=node('div',undefined,'trace-model');row.append(node('span',`Round ${e.round} LLM call · Waiting for response`),node('span',when,'muted'));debugList.append(row);models.set(e.round,row);
   title.textContent=pageCount?'Preparing an answer from retrieved materials…':'Reviewing relevant evidence…';
  }else if(e.type==='model_completed'||e.type==='model_failed'){
   const row=models.get(e.round);if(row){const usage=e.usage?.total_tokens;row.firstChild.textContent=`Round ${e.round} LLM call · ${e.type==='model_failed'?'Failed':'Responded'} · ${e.duration_seconds}s${usage!==undefined?' · '+usage+' tokens':''}`;}
  }else if(e.type==='tool_started'){
   let args;try{args=JSON.parse(e.arguments);}catch{args={invalid_arguments:e.arguments};}
   const card=node('div',undefined,'trace-tool');
   const heading=node('strong',`${toolNames[e.tool]||'Retrieve materials'} · ${recordLabel({arguments:args}).slice(0,65)}`);
   const state=node('p',e.tool==='finish_answer'?'Preparing answer…':'Retrieving…','muted');const detail=node('details');detail.append(node('summary','Record identifiers'),node('pre',JSON.stringify(args,null,2)));
   card.append(heading,state,detail);list.append(card);pending.push({tool:e.tool,round:e.round,card,state,complete:false});
  }else if(e.type==='tool_completed'||e.type==='tool_failed'){
   const step=pending.find(p=>!p.complete&&p.tool===e.tool&&p.round===e.round);if(!step)return;step.complete=true;
   const r=e.result;
   if(e.type==='tool_failed'){step.state.textContent='This record could not be retrieved. The reason is retained.';step.card.append(node('pre',r.error));step.state.className='error';return;}
   pageCount++;const detail=node('details');detail.append(node('summary','Expand retrieved content'),node('pre',r.text));step.card.append(detail);
   step.state.textContent=r.next_offset!==null?'Partial record retrieved; more content is available':'Retrieved';const meta=node('details',undefined,'technical');meta.append(node('summary','Retrieval details'),node('p',`${r.text.length.toLocaleString()} characters · ${when}`,'muted'));step.card.append(meta);
   if(r.next_offset!==null)step.card.append(node('p',`Character range ${r.offset}–${r.offset+r.text.length} / ${r.total_chars}; only this page was retrieved.`,'muted'));
   const link=node('a','View evidence →','citation');link.href=BASE+'citation/'+encodeURIComponent(r.citation_id);link.target='_blank';link.rel='noopener';step.card.append(link);
   const cited=node('p','Relating the record to the question.','muted');step.card.append(cited);pages.push({id:r.citation_id,label:cited});
   title.textContent=`Retrieving evidence · Completed ${pageCount} retrievals`;
  }else if(e.type==='answer_ready'){
   const step=pending.find(p=>!p.complete&&p.tool==='finish_answer');if(step){step.complete=true;step.state.textContent='Cited answer prepared';}
  }
 }
 function finish(answer){const used=new Set(answer.claims.flatMap(c=>c.citation_ids));for(const page of pages)page.label.textContent=used.has(page.id)?'Cited in this answer':'Retrieved but not cited in the final answer';title.textContent=`View answer evidence · ${pageCount} retrievals`;box.open=false;}
 function fail(){title.textContent=`Answer incomplete · Retained ${pageCount} retrieval records`;}
 return {event,finish,fail};
}
el('ask').addEventListener('submit',async event=>{
 event.preventDefault();if(busy||!selected())return;
 const question=el('question').value.trim();if(!question)return;
 el('conversation').append(node('article',question,'question'));const trace=makeTrace();setBusy(true);
 const start=Date.now();el('progress').textContent='Reading evidence and preparing an answer…';
 const timer=setInterval(()=>el('progress').textContent=`Retrieving and answering · ${Math.floor((Date.now()-start)/1000)} seconds`,1000);
 try{
  const response=await fetch(BASE+'api/ask',{method:'POST',headers:{'Content-Type':'application/json','Accept':'application/x-ndjson','X-QA-Token':snapshot.csrf_token},body:JSON.stringify({question,analysis_id:selected().id,session_id:sessionId})});
  if(!response.ok){const error=await response.json();throw new Error(error.error||'Request failed');}
  let answer=null;
  if(response.headers.get('Content-Type')?.includes('application/x-ndjson')){
   const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
   function consume(line){if(!line.trim())return;const e=JSON.parse(line);if(e.type==='error')throw new Error(e.error);if(e.type==='answer')answer=e.answer;else trace.event(e);}
   while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value,{stream:!done});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines)consume(line);if(done)break;}
   consume(buffer);
   if(!answer)throw new Error('Connection interrupted before the complete answer arrived. Completed calls are retained above.');
  }else{
   answer=await response.json();for(const e of answer.process_trace||[])trace.event(e);
  }
  sessionId=answer.session_id;trace.finish(answer);render(answer);el('question').value='';el('progress').textContent='Answer complete';
 }catch(error){trace.fail();el('conversation').append(node('p',error.message,'error'));el('progress').textContent='Request incomplete; you can retry';}
 finally{clearInterval(timer);setBusy(false);}
});
el('analysis').addEventListener('change',refreshAnalysis);el('reset').addEventListener('click',reset);
el('history').addEventListener('click',async()=>{
 if(busy||!selected())return;setBusy(true);
 try{
  const response=await fetch(BASE+'api/history?analysis_id='+encodeURIComponent(selected().id));
  if(!response.ok)throw new Error('Could not load saved conversations.');
  const {answers}=await response.json();if(!answers.length){el('progress').textContent='No saved conversation for this analysis';return;}
  el('conversation').replaceChildren();
  for(const answer of answers){
   el('conversation').append(node('article',answer.question,'question'));const trace=makeTrace();
   if(answer.process_trace){for(const e of answer.process_trace)trace.event(e);}
   else{for(const row of answer.tool_audit||[]){trace.event({type:'tool_started',tool:row.tool,arguments:row.arguments,round:0});trace.event({type:row.result.error?'tool_failed':'tool_completed',tool:row.tool,arguments:row.arguments,result:row.result,round:0});}}
   trace.finish(answer);render(answer);
  }
  sessionId=answers[answers.length-1].session_id;el('progress').textContent='Recent conversation restored. You can continue asking questions.';
 }catch(error){el('progress').textContent=error.message;}finally{setBusy(false);}
});
document.querySelectorAll('[data-question]').forEach(b=>b.addEventListener('click',()=>{el('question').value=b.dataset.question;el('question').focus();}));
fetch(BASE+'api/snapshot').then(r=>{if(!r.ok)throw new Error('Could not load the snapshot');return r.json();}).then(data=>{snapshot=data;el('case').textContent=data.case_id.replace('ma_','Study ');el('study-question').textContent=data.topic||'Explore the pooled result, understand study selection, and trace the original sources.';if(data.library_url){el('library-link').hidden=false;el('library-link').href=data.library_url;}el('snapshot').textContent=`${data.documents} articles · ${data.analyses.length} analyses`;for(const a of data.analyses){const o=node('option',a.title.replace(/_/g,' '));o.value=a.id;el('analysis').append(o);}if(data.warnings.length){el('warning-box').hidden=false;for(const w of data.warnings)el('warnings').append(node('li',w));}const requested=new URLSearchParams(location.search).get('analysis');if(requested&&data.analyses.some(a=>a.id===requested))el('analysis').value=requested;refreshAnalysis();if(!data.analyses.length)el('submit').disabled=true;}).catch(error=>{el('case').textContent=error.message;el('submit').disabled=true;});

const languageLink=document.getElementById('language-toggle');if(languageLink){const u=new URL(location.href);u.searchParams.set('lang',document.documentElement.lang.startsWith('zh')?'en':'zh');languageLink.href=u.href;}
