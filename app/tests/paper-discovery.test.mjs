import {test} from 'node:test';import assert from 'node:assert/strict';
import {filterDiscovery,relatedPapers} from '../lib/paper-discovery.ts';
import {mergeArchive} from '../lib/atlas-archive.ts';
const papers=[{id:'a',title:'Alpha',collection:'iclr/2027',keywords:['Robotics'],primary_area:'RL',tldr:'Learning robot policies',number:2,updated:1},{id:'b',title:'Beta',collection:'iclr/2027',keywords:['Robotics','Control'],primary_area:'RL',abstract_preview:'Grounded planning',number:1,updated:2},{id:'c',title:'Gamma',collection:'iclr/2026',keywords:['Control'],primary_area:'Theory'}];
const base={q:'',collection:'iclr/2027',area:'',tag:'',saved:false,sort:'title'};
test('search combines words across metadata and composes with area, keyword and saved filters',()=>{
 assert.deepEqual(filterDiscovery(papers,{...base,q:'robot policies'},new Set()).map(p=>p.id),['a']);
 assert.deepEqual(filterDiscovery(papers,{...base,area:'RL',tag:'robotics',saved:true},new Set(['b'])).map(p=>p.id),['b']);
 assert.deepEqual(filterDiscovery(papers,{...base,sort:'number'},new Set()).map(p=>p.id),['b','a']);
 assert.deepEqual(filterDiscovery(papers,{...base,q:'grounded'},new Set()).map(p=>p.id),['b']);
 assert.deepEqual(relatedPapers(papers[0],papers).map(p=>p.id),['b']);
});
test('enriches classified papers without overwriting reports or mutating source',()=>{
 const hash='a'.repeat(64),p={id:hash.slice(0,24),title:'Title',pdf_key:`papers/${hash}.pdf`,classified:true,models:['v8'],detail_key:'verified'};
 const result=mergeArchive([p],[],[{key:p.pdf_key,forum_id:'forum',keywords:['Control']}]);
 assert.equal(result[0].classified,true);assert.equal(result[0].detail_key,'verified');assert.equal(result[0].forum_id,'forum');assert.equal(p.forum_id,undefined);
});
