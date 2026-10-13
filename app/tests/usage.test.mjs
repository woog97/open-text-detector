import {test} from 'node:test';
import assert from 'node:assert/strict';
import {classify,device,browser,recordUsage} from '../atlas-cloud/usage.ts';
const u=s=>new URL('https://opentextdetector.com'+s);
test('counts pages, papers opened, list queries and feedback; skips PDF bytes, metadata, prefetch and assets',()=>{
 assert.deepEqual(classify(u('/'),'GET'),{event:'page',item:'home',detail:''});
 assert.equal(classify(u('/?view=calibration'),'GET').item,'calibration');
 assert.equal(classify(u('/?collection=iclr%2F2027&paper=abc'),'GET').item,'reader');
 assert.equal(classify(u('/?collection=iclr%2F2027'),'GET').item,'submissions');
 assert.deepEqual(classify(u('/backend/v1/pdf-reader/'+'a'.repeat(24)),'GET'),{event:'paper',item:'a'.repeat(24),detail:''});
 const b=classify(u('/backend/v1/browse?collection=iclr%2F2027&sort=score&q=Jane+Doe&area=x'),'GET');
 assert.equal(b.event,'browse');assert.match(b.detail,/sort=score/);assert.match(b.detail,/search/);assert.doesNotMatch(b.detail,/Jane/,'search text is never stored');
 assert.equal(classify(u('/backend/v1/feedback'),'POST').event,'feedback');
 for(const s of ['/backend/v1/pdf-reader/'+'a'.repeat(24)+'/file','/backend/v1/pdf-reader/'+'a'.repeat(24)+'/metadata','/backend/v1/browse?paper=x','/robots.txt','/assets/x.js'])assert.equal(classify(u(s),'GET'),null,s);
 assert.equal(classify(u('/'),'POST'),null);
});
test('device and browser from the user agent; bots are labelled',()=>{
 assert.equal(device('Mozilla/5.0 (iPhone; CPU iPhone OS 18_0) Mobile/15E148 Safari/604.1'),'mobile');
 assert.equal(device('Mozilla/5.0 (Macintosh) AppleWebKit Chrome/140 Safari/537.36'),'desktop');
 assert.equal(device('Mozilla/5.0 (compatible; GPTBot/1.2)'),'bot');assert.equal(device('python-httpx/0.28'),'bot');assert.equal(device(''),'bot');
 assert.equal(device('Mozilla/5.0 Chrome/140',{verifiedBotCategory:'Search Engine Crawler'}),'bot');
 assert.equal(browser('Mozilla/5.0 Chrome/140 Safari/537.36 Edg/140'),'Edge');assert.equal(browser('Mozilla/5.0 Firefox/141.0'),'Firefox');
});
test('a data point has the fields, never the IP address, and the visitor hash changes by day',async()=>{
 const points=[],sink={writeDataPoint:p=>points.push(p)};
 const req=()=>{const r=new Request('https://opentextdetector.com/?view=data',{headers:{'CF-Connecting-IP':'198.51.100.7','User-Agent':'Mozilla/5.0 Chrome/140','Referer':'https://news.ycombinator.com/item?id=1','Accept-Language':'en-GB,en;q=0.9'}});
  Object.defineProperty(r,'cf',{value:{country:'GB',city:'Cambridge',region:'England',asOrganization:'University of Cambridge'}});return r;};
 await recordUsage(sink,'salt',req(),200,new Date('2026-10-13T12:00:00Z'));
 await recordUsage(sink,'salt',req(),200,new Date('2026-10-13T20:00:00Z'));
 await recordUsage(sink,'salt',req(),200,new Date('2026-10-14T12:00:00Z'));
 const b=points[0].blobs;
 assert.deepEqual(points[0].indexes,['page']);assert.deepEqual(b.slice(0,9),['page','data','GB','England / Cambridge','University of Cambridge','desktop','Chrome','news.ycombinator.com','en-GB']);
 assert.equal(points[0].blobs[9],points[1].blobs[9],'same visitor, same day');assert.notEqual(points[0].blobs[9],points[2].blobs[9],'new day, new hash');
 assert.doesNotMatch(JSON.stringify(points),/198\.51\.100\.7/);
 await recordUsage({writeDataPoint(){throw new Error('down')}},'salt',req(),200);  // never throws
 await recordUsage(undefined,'salt',req(),200);
});
