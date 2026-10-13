/** Validate the actual browser parser and PDF mapping on every staged Laya report. */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import assert from 'node:assert/strict';
import {validateReaderDetail} from '../app/lib/pdf-reader-data.ts';
import {supportsSavedPositions,positionPassages} from '../app/lib/pdf-positions.ts';
import {summarizePassages} from '../app/lib/paper-scores.ts';
const root=path.resolve(import.meta.dirname,'../research/classifications/laya-atlas-five-per-year');
const records=fs.readdirSync(path.join(root,'staged')).filter(n=>n.endsWith('.json'));
let phrases=0,mapped=0;const papers=[];
for(const filename of records){
 const record=JSON.parse(fs.readFileSync(path.join(root,'staged',filename),'utf8'));
 const raw=zlib.gunzipSync(fs.readFileSync(path.join(root,'objects',path.basename(record.item.detail_key))));
 const detail=validateReaderDetail(JSON.parse(raw));
 const report=detail.reports.find(r=>r.model.id==='laya');assert.ok(report);
 const map=detail.position_maps[report.text_sha256];assert.ok(supportsPositions(map));
 const positioned=positionPassages(report.text,report.result.segments,map);
 const count=positioned.filter(s=>s.mapped).length;phrases+=positioned.length;mapped+=count;
 assert.deepEqual(summarizePassages(report.result.segments,report.text),record.item.score_summaries.laya);
 assert.ok(count>0);
 papers.push({id:record.item.id,detail_key:record.item.detail_key,phrases:positioned.length,mapped:count});
}
function supportsPositions(map){return supportsSavedPositions(map);}
const result={papers:records.length,phrases,mapped,all_browser_reports_valid:true,paper_results:papers};
fs.writeFileSync(path.join(root,'browser-data-audit.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify({papers:records.length,phrases,mapped,all_browser_reports_valid:true}));
