// The site moved to a new Worker when the project was renamed to Open Text Detector (October 11, 2026).
// These checks keep every place that names the Worker, its address or its bucket in agreement, and check that the
// old address redirects to the same path on the new one.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {redirect,NEW_ORIGIN} from '../atlas-cloud/redirect-worker.ts';

const app=new URL('../',import.meta.url),repo=new URL('../../',import.meta.url);
const json=path=>JSON.parse(readFileSync(new URL(path,app),'utf8'));
const text=path=>readFileSync(new URL(path,repo),'utf8');
const site=json('wrangler.atlas.json'),old=json('wrangler.atlas-redirect.json');
const origin=name=>`https://${name}.woog09.workers.dev`;

test('the site runs on the open-text-detector Worker and keeps the existing R2 bucket',()=>{
 assert.equal(site.name,'open-text-detector');
 assert.deepEqual(site.r2_buckets,[{binding:'BUCKET',bucket_name:'pangram-paper-atlas'}]); // bucket names do not change
 assert.equal(site.main,'atlas-cloud/worker.ts');
});

test('the old Worker name only redirects: no bucket, no assets, same account',()=>{
 assert.equal(old.name,'pangram-paper-atlas');
 assert.equal(old.account_id,site.account_id);
 assert.equal(old.main,'atlas-cloud/redirect-worker.ts');
 assert.ok(existsSync(new URL(old.main,app)));
 assert.equal(old.r2_buckets,undefined);
 assert.equal(old.assets,undefined);
});

test('every place that calls the live site uses the new Worker address',()=>{
 const want=origin(site.name);
 assert.equal(NEW_ORIGIN,want);
 assert.match(text('browser-extension/src/site.ts'),new RegExp(`SITE = '${want}'`));
 assert.ok(JSON.parse(text('browser-extension/manifest.json')).host_permissions.includes(want+'/*'));
 assert.ok(text('app/scripts/preview-atlas.mjs').includes(`"${want}"`));
 assert.ok(text('app/atlas-local/worker.ts').includes(`'${want}'`));
});

test('GET and HEAD on the old address get a 301 to the same path and query',async()=>{
 for(const method of ['GET','HEAD']){
  const r=redirect(new Request(origin(old.name)+'/?collection=iclr%2F2027&paper=91f1ad507a924a58f9a3c1d7',{method}));
  assert.equal(r.status,301);
  assert.equal(r.headers.get('location'),NEW_ORIGIN+'/?collection=iclr%2F2027&paper=91f1ad507a924a58f9a3c1d7');
 }
 const api=redirect(new Request(origin(old.name)+'/backend/v1/browse?q=a%20b&collection='));
 assert.equal(api.headers.get('location'),NEW_ORIGIN+'/backend/v1/browse?q=a%20b&collection=');
 assert.equal(redirect(new Request(origin(old.name))).headers.get('location'),NEW_ORIGIN+'/');
});

test('other methods get a 308, which keeps the method and body',async()=>{
 const r=redirect(new Request(origin(old.name)+'/backend/upload',{method:'POST',body:'x'}));
 assert.equal(r.status,308);
 assert.equal(r.headers.get('location'),NEW_ORIGIN+'/backend/upload');
});

test('the redirect is cached for at most an hour, so a rollback takes effect quickly',()=>{
 assert.equal(redirect(new Request(origin(old.name)+'/')).headers.get('cache-control'),'public, max-age=3600');
});
