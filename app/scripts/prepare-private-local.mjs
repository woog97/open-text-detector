import {readFileSync,writeFileSync,existsSync,chmodSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
const app=fileURLToPath(new URL('../',import.meta.url));
const data=path.resolve(app,'../private-results/20260926');
const auth=readFileSync(path.join(data,'auth-secret.json'),'utf8');
JSON.parse(auth);
const vars=path.join(app,'atlas-local/.dev.vars');
writeFileSync(vars,`PRIVATE_AUTH='${auth}'\n`,{mode:0o600});chmodSync(vars,0o600);
const catalogue=JSON.parse(readFileSync(path.join(data,'upload/catalogue.json'),'utf8'));
const files=['catalogue.json',...catalogue.items.flatMap(p=>[`papers/${p.id}/paper.pdf`,`papers/${p.id}/detail.json.gz`])];
const fingerprint=createHash('sha256');for(const file of files)fingerprint.update(readFileSync(path.join(data,'upload',file)));
const signature=fingerprint.digest('hex'),stamp=path.join(data,'.local-seed');
if(!existsSync(stamp)||readFileSync(stamp,'utf8')!==signature){
 for(const file of files){
  const result=spawnSync(process.execPath,[path.join(app,'node_modules/wrangler/bin/wrangler.js'),'r2','object','put',`open-text-detector-private-results-local/${file}`,'--file',path.join(data,'upload',file),'--local','--config',path.join(app,'atlas-local/wrangler.json'),'--persist-to',path.join(data,'.local-storage')],{cwd:app,encoding:'utf8',env:{...process.env,WRANGLER_SEND_METRICS:'false'}});
  if(result.status!==0){process.stderr.write(result.stderr||result.stdout);process.exit(result.status||1);}
  console.log('Prepared local object:',file);
 }
 writeFileSync(stamp,signature);
}
console.log('Private local storage ready. No private data uploaded.');
