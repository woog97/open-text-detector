// Builds the unpacked extension into dist/ with the app's esbuild and pdfjs-dist (run `npm ci` in app/ first).
import {cpSync, mkdirSync, rmSync} from 'node:fs';
import {createRequire} from 'node:module';
const require = createRequire(new URL('../app/package.json', import.meta.url));
const esbuild = require('esbuild');
const here = new URL('.', import.meta.url).pathname, dist = `${here}dist`;
const pdfjs = new URL('../app/node_modules/pdfjs-dist/', import.meta.url).pathname;

rmSync(dist, {recursive: true, force: true});
mkdirSync(`${dist}/pdfjs`, {recursive: true});
cpSync(`${here}manifest.json`, `${dist}/manifest.json`);
cpSync(`${here}static`, dist, {recursive: true});
cpSync(`${pdfjs}build/pdf.worker.min.mjs`, `${dist}/pdfjs/pdf.worker.min.mjs`);
cpSync(`${pdfjs}web/pdf_viewer.css`, `${dist}/pdfjs/pdf_viewer.css`);
for (const folder of ['cmaps', 'standard_fonts', 'wasm']) cpSync(`${pdfjs}${folder}`, `${dist}/pdfjs/${folder}`, {recursive: true});

const common = {bundle: true, target: 'chrome120', nodePaths: [`${here}../app/node_modules`], logLevel: 'info', legalComments: 'none'};
await esbuild.build({...common, entryPoints: {background: `${here}src/background.ts`, viewer: `${here}src/viewer.ts`, popup: `${here}src/popup.ts`}, outdir: dist, format: 'esm'});
// Content scripts cannot be modules.
await esbuild.build({...common, entryPoints: {content: `${here}src/content.ts`}, outdir: dist, format: 'iife'});
