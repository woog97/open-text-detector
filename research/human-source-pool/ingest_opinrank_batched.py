"""OpinRank batching optimization; identical source sampling and candidate records."""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile
from collect_pool import BINS,LENGTH_WEIGHTS,apportion,atomic_json,digest,now
from expand_pool import connect
from ingest_opinrank import ARCHIVE_SHA,DATASET,diversified_rows,pair


def prepare(record,raw):
    # Compression and JSON serialization are CPU work and do not hold the SQLite writer lock.
    return (record, digest(' '.join(record['text'].casefold().split())),
            json.dumps(record,ensure_ascii=False), gzip.compress(json.dumps(raw,ensure_ascii=False).encode()))


def flush(db,prepared,quota,targets,cursor_key,scanned):
    """Atomically append one bounded batch plus its scan position; interrupted batches roll back."""
    reasons=Counter();db.execute('BEGIN IMMEDIATE')
    try:
        bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='opinrank' GROUP BY bin"))
        count=sum(bins.values())
        for row,norm,payload,compressed in prepared:
            if count>=quota:break
            bin_id=row['length_bin']
            if bins.get(bin_id,0)>=targets[str(bin_id)]:
                reasons['length_bin_full']+=1;continue
            inserted=db.execute('INSERT OR IGNORE INTO passages VALUES (?,?,?,?,?,?)',
                (row['record_id'],norm,'opinrank',row['parent_document_id'],bin_id,payload)).rowcount
            if inserted:
                db.execute('INSERT OR IGNORE INTO documents VALUES (?,?)',(row['raw_text_sha256'],compressed))
                bins[bin_id]=bins.get(bin_id,0)+1;count+=1;reasons['accepted']+=1
            else:reasons['duplicate']+=1
        db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)',('opinrank',cursor_key,scanned,0))
        db.commit()
        return count,bins,reasons
    except BaseException:
        db.rollback();raise


def collect(base,batch_size=100):
    started=time.monotonic();directory=base/'source-downloads/opinrank'
    manifest=json.loads((directory/'manifest.json').read_text());path=directory/manifest['archive']
    if manifest['sha256']!=ARCHIVE_SHA or hashlib.sha256(path.read_bytes()).hexdigest()!=ARCHIVE_SHA:
        raise RuntimeError('OpinRank archive checksum mismatch')
    plan=json.loads((base/'pipeline/sampling-plan.json').read_text())
    quota=next(s['planned_passages'] for s in plan['source_quotas'] if s['source_id']=='opinrank')
    targets=apportion(quota,{str(i):n for i,n in enumerate(LENGTH_WEIGHTS['reviews'])})
    cursor_key='opinrank-batched-v1:'+ARCHIVE_SHA+':'+str(quota)
    with (base/'opinrank.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=connect(base/'collection.sqlite3');seen=Counter();pending=[];scanned=0;batch_scanned=0;commits=0
        bins=dict(db.execute("SELECT bin,count(*) FROM passages WHERE source='opinrank' GROUP BY bin"));count=sum(bins.values());initial=count
        old=db.execute('SELECT position FROM cursors WHERE source=? AND file=?',('opinrank',cursor_key)).fetchone()
        last=old[0] if old else 0
        def status(state):
            return {'source_id':'opinrank','count':count,'target':quota,'scanned':scanned,'state':state,'updated_at':now(),
                    'reasons':dict(seen),'batch_size':batch_size,'commits_this_run':commits,'all_quarantined':True,
                    'new_candidates':count-initial,'elapsed_seconds':round(time.monotonic()-started,3)}
        try:
            with zipfile.ZipFile(path) as archive:
                for scanned,row in enumerate(diversified_rows(archive),1):
                    if count>=quota:break
                    if scanned<=last:continue
                    batch_scanned+=1
                    wc=len(row['text'].split());bin_id=next((i for i,(lo,hi) in enumerate(BINS) if lo<=wc<=hi),None)
                    # Full bins cannot become underfilled while this source lock is held.
                    if bin_id is not None and bins.get(bin_id,0)>=targets[str(bin_id)]:
                        seen['length_bin_full']+=1
                    else:
                        result=pair(row,manifest)
                        if result:pending.append(prepare(*result))
                        else:seen['quality_or_length']+=1
                    if batch_scanned>=batch_size:
                        count,bins,delta=flush(db,pending,quota,targets,cursor_key,scanned);seen.update(delta)
                        pending=[];batch_scanned=0;commits+=1
                        if commits%5==0:atomic_json(base/'progress/opinrank.json',status('collecting'))
                if batch_scanned:
                    count,bins,delta=flush(db,pending,quota,targets,cursor_key,scanned);seen.update(delta);commits+=1
            groups=Counter()
            for (payload,) in db.execute("SELECT row FROM passages WHERE source='opinrank'"):
                r=json.loads(payload);groups[r['review_domain']+'/'+r['review_group']]+=1
            result=status('quota_filled' if count==quota else 'archive_exhausted');result['groups']=dict(groups)
            atomic_json(base/'progress/opinrank.json',result);print(json.dumps(result),flush=True);return result
        finally:db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=100);args=parser.parse_args()
    if not 1<=args.batch_size<=100:parser.error('batch-size must be 1–100')
    collect(args.base,args.batch_size)

if __name__=='__main__':main()
