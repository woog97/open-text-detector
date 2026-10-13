import {test} from 'node:test';
import assert from 'node:assert/strict';
import {needsReaderReload} from '../lib/reader-errors.ts';
test('stale deployment chunks require a full reload, ordinary reader errors allow retry',()=>{
 for(const text of ['Failed to fetch dynamically imported module: /assets/old.js','Importing a module script failed.','error loading dynamically imported module','Unable to preload CSS for /assets/old.css'])assert.equal(needsReaderReload(new Error(text)),true);
 assert.equal(needsReaderReload(new Error('Unexpected document shape')),false);
 assert.equal(needsReaderReload(null),false);
});
