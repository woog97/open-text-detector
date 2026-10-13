"""Verify all replacements, update references, then remove redundant extraction files."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from open_text_detector_backend.result_codec import decode
from open_text_detector_backend.sqlite_runtime import sqlite3
OUT=ROOT/'research/extractions/positioned'
def main():
    rows={r['pdf_sha256']:r for r in map(json.loads,(OUT/'migration.jsonl').read_text().splitlines())}
    plan=json.loads((OUT/'plan.json').read_text());assert len(rows)==len(plan),'Migration is not finished'
    conn=sqlite3.connect(f'file:{ROOT}/backend/.data/workspace.sqlite3?mode=ro',uri=True)
    protected={hashlib.sha256(r[0].encode()).hexdigest() for r in conn.execute("SELECT text FROM scans WHERE kind='text'")}
    conn.close()
    aliases={};canonical={};bytes_new=0;unused=[]
    for r in rows.values():
        canonical[r['pdf_sha256']]=r['canonical']['path']
        for a in r['artifacts']:
            blob=(ROOT/a['path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==a['blob_sha256']
            obj=decode(blob);assert obj['pdf_sha256']==r['pdf_sha256'];assert hashlib.sha256(obj['text'].encode()).hexdigest()==obj['text_sha256'];bytes_new+=len(blob)
            for box in obj['rectangles']:
                assert 0<=box['start']<box['end']<=len(obj['text'])
                assert 1<=box['page']<=len(obj['pages'])
                assert box['x0']<=box['x1'] and box['y0']<=box['y1']
        for a in r['aliases']:
            p=ROOT/a['old_path'];replacement=decode((ROOT/a['replacement']).read_bytes())
            if p.exists():
                assert hashlib.sha256(p.read_bytes()).hexdigest()==a['old_file_sha256'],str(p)
                original=hashlib.sha256(p.read_text().encode()).hexdigest()
                assert original==a.get('original_text_sha256',a['text_sha256'])
                if original in protected:assert p.read_text()==replacement['text'],str(p)
            original=a.get('original_text_sha256',a['text_sha256'])
            if original not in protected:
                a={**a,'original_text_sha256':original,'text_sha256':r['canonical']['text_sha256'],'replacement':r['canonical']['path'],'preserve_exact_text':False}
            aliases[a['old_path']]=a
        retained=[]
        for a in r['artifacts']:
            if a['kind']=='canonical' or a['text_sha256'] in protected:retained.append(a)
            else:unused.append(ROOT/a['path']);bytes_new-=a['bytes']
        r['artifacts']=retained
        r['aliases']=[aliases[a['old_path']] for a in r['aliases']]
    retained_paths={a['path'] for r in rows.values() for a in r['artifacts']}
    for p in (OUT/'objects').rglob('*.pgf'):
        if str(p.relative_to(ROOT)) not in retained_paths and p not in unused:
            obj=decode(p.read_bytes())
            assert obj['pdf_sha256'] in canonical and obj['text_sha256'] not in protected
            unused.append(p)
    # Additional verbose coordinate dumps are superseded by canonical positioned objects.
    redundant=[]
    benchmark=ROOT/'research/extractions/iclr2025-benchmark'
    for r in json.loads((benchmark/'results.json').read_text())['papers']:
        assert r['sha256'] in canonical
        p=benchmark/'positioned'/(r['id']+'.xhtml')
        if p.exists():redundant.append(p)
    hyper=ROOT/'research/extractions/hyperdas'
    if (hyper/'positioned-text.json').exists():
        h=json.loads((hyper/'positioned-text.json').read_text())['pdf_sha256'];assert h in canonical
        redundant.extend(p for p in [hyper/'positioned-text.json',hyper/'positioned-text.xhtml'] if p.exists())
    def rewrite(obj):
        if isinstance(obj,list):return [rewrite(x) for x in obj]
        if not isinstance(obj,dict):return obj
        obj={k:rewrite(v) for k,v in obj.items()}
        source=obj.get('text_file')
        if source in aliases:
            a=aliases[source];obj['text_file']=a['replacement'];obj['text_sha256']=a['text_sha256'];obj['text_format']='pgf1-positioned-text-v1'
            obj['text_file_sha256']=hashlib.sha256((ROOT/a['replacement']).read_bytes()).hexdigest()
            if a.get('preserve_exact_text') is False:
                obj['previous_text_sha256']=a['original_text_sha256']
                text=decode((ROOT/a['replacement']).read_bytes())['text']
                if 'words' in obj:obj['words']=len(text.split())
                if 'characters' in obj:obj['characters']=len(text)
        if 'native_text_file' in obj and obj['native_text_file'] in aliases:
            obj['native_text_file']=None
            obj['empty_native_output_removed']=True
        return obj
    changed=[]
    for base in [ROOT/'research/classifications',ROOT/'research/extractions']:
        for p in list(base.rglob('*.json'))+list(base.rglob('*.jsonl')):
            if OUT in p.parents or p in redundant:continue
            if p.suffix=='.jsonl':
                old=[json.loads(x) for x in p.read_text().splitlines()];new=rewrite(old)
                content=''.join(json.dumps(x)+'\n' for x in new)
            else:
                old=json.loads(p.read_text());new=rewrite(old);content=json.dumps(new,indent=2)
            if old!=new:
                temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(content);temp.replace(p);changed.append(str(p.relative_to(ROOT)))
    candidates=[ROOT/p for p in aliases if (ROOT/p).exists()]+redundant+[p for p in unused if p.exists()]
    candidates=list(dict.fromkeys(candidates))
    for p in candidates:assert p.resolve().is_relative_to((ROOT/'research/extractions').resolve())
    deletion=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in candidates]
    report={'status':'verified_pending_cleanup','verified_pdfs':len(rows),'compressed_bytes':bytes_new,'references_updated':changed,'deleted_files':deletion,'bytes_removed':sum(r['bytes'] for r in deletion)}
    (OUT/'cleanup-ledger.json').write_text(json.dumps(report,indent=2))
    index=sqlite3.connect(OUT/'index.sqlite3')
    for p in unused:index.execute('DELETE FROM artifacts WHERE path=?',(str(p.relative_to(ROOT)),))
    index.commit();index.close()
    temp=OUT/'migration.final.jsonl';temp.write_text(''.join(json.dumps(r)+'\n' for r in rows.values()));temp.replace(OUT/'migration.jsonl')
    for p in candidates:p.unlink()
    report['status']='completed'
    (OUT/'cleanup-ledger.json').write_text(json.dumps(report,indent=2))
    (OUT/'summary.json').write_text(json.dumps({'papers':len(rows),'artifacts':sum(len(r['artifacts']) for r in rows.values()),'compressed_bytes':bytes_new,'redundant_files_removed':len(candidates),'bytes_removed':report['bytes_removed']},indent=2))
    print({k:v for k,v in report.items() if k not in ('references_updated','deleted_files')});print('Removed',len(candidates),'verified redundant files.')
if __name__=='__main__':main()
