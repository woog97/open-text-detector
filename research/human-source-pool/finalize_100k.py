"""Audit and publish the exact 100k remote candidate pool after bounded collection."""
from pathlib import Path
import sys,json,sqlite3,hashlib,time,requests,subprocess,os,collections
from urllib.parse import quote
from huggingface_hub import HfApi
from collect_pool import atomic_json,now
B=Path('/tmp/pangram-human-active-20261002');P=B/'pipeline';R=B/'finish100k';token=sys.stdin.readline().strip();assert token
state=json.loads((R/'status.json').read_text());assert state['state']=='workers_finished' and all(v==0 for v in state['workers'].values()),state
receipt=json.loads((R/'checkpoint.json').read_text());dbpath=Path(receipt['database']);assert receipt['candidate_total']==100000
assert hashlib.sha256(dbpath.read_bytes()).hexdigest()==receipt['sha256']
d=sqlite3.connect('file:'+str(dbpath)+'?mode=ro&immutable=1',uri=True);assert d.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
plan=json.loads((P/'sampling-plan.json').read_text());quotas={r['source_id']:r['planned_passages'] for r in plan['source_quotas']};counts=dict(d.execute('SELECT source,count(*) FROM passages GROUP BY source'));assert sum(counts.values())==100000 and all(counts.get(s,0)==q for s,q in quotas.items())
old=B/'checkpoints/finish100k-progress-20261002T192425Z.sqlite3';d.execute('ATTACH DATABASE ? AS prior',('file:'+str(old)+'?mode=ro&immutable=1',));assert d.execute('SELECT count(*) FROM prior.passages p LEFT JOIN main.passages n ON p.id=n.id WHERE n.id IS NULL OR n.row!=p.row').fetchone()[0]==0
newcounts=collections.Counter();protected=0;categories=collections.Counter();admitted=0
from ingest_asap2 import protected_prompt
for source,saved in d.execute('SELECT source,row FROM main.passages'):
 row=json.loads(saved);categories[row['category']]+=1;admitted+=int(row.get('training_eligible',False) or row.get('admission_status')!='quarantined_candidate')
assert admitted==0 and dict(categories)==plan['category_counts']
for source,saved in d.execute('SELECT n.source,n.row FROM main.passages n LEFT JOIN prior.passages p ON p.id=n.id WHERE p.id IS NULL'):
 row=json.loads(saved);newcounts[source]+=1;assert source in ['asap2','gutenberg','foodista','voa']
 if source=='asap2':assert not protected_prompt({'prompt_name':row['title']})
assert dict(newcounts)=={'asap2':1263,'foodista':724,'gutenberg':2551,'voa':65}
assert d.execute('SELECT count(*) FROM passages').fetchone()[0]==d.execute('SELECT count(DISTINCT norm) FROM passages').fetchone()[0]
d.close()
migration=json.loads((R/'migration/receipt.json').read_text());original=migration['source_quotas_before'];distribution={'original_target_total':100000,'actual_total':100000,'category_totals_unchanged':True,'sources':[{'source_id':s,'original_target':q,'revised_target':quotas[s],'actual':counts.get(s,0),'difference_from_original':counts.get(s,0)-q} for s,q in original.items()],'user_authorized_reallocation':migration['user_authorization']}
atomic_json(P/'original-distribution-report.json',distribution);atomic_json(P/'quota-reallocation-receipt.json',migration)
audit={'candidate_total':100000,'admitted_total':0,'global_exact_dedup_verified':True,'all_prior_95397_passages_byte_identical':True,'new_passages_by_source':dict(newcounts),'new_asap_protected_prompts_excluded':True,'category_totals_unchanged':True,'database':str(dbpath),'sha256':receipt['sha256'],'at':now(),'remaining_admission_caveat':'All records remain quarantined candidates; earlier877ASAP protected-family flags retained and not admitted. Full training admission/near-duplicate audit is separate.'};atomic_json(P/'finish100k-audit.json',audit)
api=HfApi(token=token);prefix='workspace/human-source-mix-v2-recovered-20261002/';bucket='open-text-detector/training-storage'
# Full final durable database readback, not merely committed metadata.
r=requests.get('https://huggingface.co/buckets/'+bucket+'/resolve/'+quote(receipt['bucket_key'],safe=''),headers={'Authorization':'Bearer '+token,'Accept-Encoding':'identity'},stream=True,timeout=(15,90));r.raise_for_status();h=hashlib.sha256()
for chunk in r.iter_content(4*1024*1024):h.update(chunk)
assert h.hexdigest()==receipt['sha256'];audit['checkpoint_full_get_sha256_verified']=True;atomic_json(P/'finish100k-audit.json',audit)
api.batch_bucket_files(bucket,add=[(P/n,prefix+'pipeline/'+n) for n in ['original-distribution-report.json','quota-reallocation-receipt.json','finish100k-audit.json']]);print('FINAL_AUDIT',json.dumps(audit),flush=True)
name='human-100000-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
with (B/'snapshots'/(name+'.log')).open('a') as log:
 child=subprocess.Popen([sys.executable,str(P/'publish_snapshot.py'),'--base',str(B),'--name',name,'--repo-id','open-text-detector/human-source-mix-v1','--database',str(dbpath)],stdin=subprocess.PIPE,stdout=log,stderr=log,start_new_session=True,cwd=P,env=dict(os.environ))
child.stdin.write((token+'\n').encode());child.stdin.close();info={'pid':child.pid,'start_ticks':Path('/proc',str(child.pid),'stat').read_text().split()[21],'snapshot':str(B/'snapshots'/name),'database':str(dbpath),'base':str(B)};atomic_json(B/'finish100k-final-publication-process.json',info);print('PUBLICATION',json.dumps(info),flush=True)
