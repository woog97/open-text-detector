"""Build an additive, versioned clean-text dataset from existing Space PGF sidecars.

Run on the training Space. Original PDFs, Parquet files, text and coordinates are
read-only. Supports a sample-ID gate before full processing. No model inference.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from open_text_detector_backend.text_cleanup import clean, VERSION, validate
from open_text_detector_backend.result_codec import encode, decode


def digest(raw):return hashlib.sha256(raw).hexdigest()


def atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def one(job):
    root,paper,output,regions_dir=job;root=Path(root);output=Path(output)
    source=root/paper.get('base','source')/paper['text_file'];raw=source.read_bytes()
    if digest(raw)!=paper['blob_sha256']:raise ValueError('Source blob checksum mismatch')
    original=decode(raw)
    if original['pdf_sha256']!=paper['pdf_sha256'] or original['text_sha256']!=paper['text_sha256']:
        raise ValueError('Source identity mismatch')
    regions=None
    if regions_dir:
        import zstandard
        from open_text_detector_backend.text_cleanup import mineru_figure_regions
        layout=Path(regions_dir)/f"{paper['pdf_sha256']}.middle.json.zst"
        regions=mineru_figure_regions(json.loads(zstandard.ZstdDecompressor().decompress(layout.read_bytes(),max_output_size=2**30)))
    result=clean(original,figure_regions=regions)
    packed=encode(result,level=9)
    if decode(packed)!=result:raise ValueError('Clean artifact roundtrip mismatch')
    sidecar=output/'objects'/paper['pdf_sha256']/(paper['text_sha256']+'.pgf')
    if sidecar.exists():
        if sidecar.read_bytes()!=packed:raise ValueError('Existing derived artifact differs')
    else:atomic(sidecar,packed)
    return {'id':paper['forum_id'],**paper.get('extra',{}),'pdf_sha256':paper['pdf_sha256'],
            'text':result['text'],'text_sha256':result['text_sha256'],
            'source_text_sha256':result['source_text_sha256'],'source_blob_sha256':paper['blob_sha256'],
            'source_dataset':root.name,'source_positions_path':paper['text_file'],
            'clean_positions_path':str(sidecar.relative_to(output)),
            'clean_blob_sha256':digest(packed),'cleanup_version':result['format'],
            'stats_json':json.dumps(result['stats'],sort_keys=True),
            'flag_count':len(result['flags']),'pages':len(original['pages'])}


def source_papers(root,inputs):
    """Round datasets list papers in transfer-plan.json; the complete mirror
    lists them in its Parquet shards with content-addressed position sidecars
    (objects/<sha256>.pgf). Rows without an archived extraction are skipped."""
    if (root/'transfer-plan.json').exists():
        raw=(root/'transfer-plan.json').read_bytes()
        inputs.append({'source_dataset':root.name,'manifest_sha256':digest(raw)})
        return json.loads(raw)['papers']
    import pyarrow.parquet as pq
    papers=[];shards=sorted((root/'data').glob('train-*.parquet'))
    for shard in shards:
        names=set(pq.read_schema(shard).names)
        extra=[c for c in ('title','conference','year','forum_id') if c in names]
        columns=['id','pdf_sha256','text_sha256','positions_path']+(['blob_sha256'] if 'blob_sha256' in names else [])+extra
        for row in pq.read_table(shard,columns=columns).to_pylist():
            if not row['positions_path']:continue
            papers.append({'forum_id':row['id'],'pdf_sha256':row['pdf_sha256'],'text_sha256':row['text_sha256'],
                           'text_file':row['positions_path'],'base':'.',
                           'blob_sha256':row.get('blob_sha256') or Path(row['positions_path']).stem,
                           'extra':{c:row[c] for c in extra}})
    inputs.append({'source_dataset':root.name,'manifest_sha256':digest(b''.join(digest(s.read_bytes()).encode() for s in shards))})
    return papers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',action='append',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--sample-ids',type=Path)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--figure-regions',type=Path,help='MinerU objects dir (<pdf_sha256>.middle.json.zst); only papers with layout data are processed')
    args=p.parse_args()
    if not 1<=args.workers<=(os.cpu_count() or 1):p.error('Too many workers for this machine')
    output=args.output.resolve();sources=[s.resolve() for s in args.source]
    if any(output==s or output.is_relative_to(s) or s.is_relative_to(output) for s in sources):
        p.error('Output and source directories must be disjoint')
    from open_text_detector_backend import text_cleanup
    code_hash=digest(Path(text_cleanup.__file__).read_bytes())
    wanted=None
    if args.sample_ids:
        selected=json.loads(args.sample_ids.read_text())
        wanted=set(selected['sample_ids'] if isinstance(selected,dict) else selected)
    jobs=[];inputs=[]
    for root in sources:
        for paper in source_papers(root,inputs):
            if args.figure_regions and not (args.figure_regions/f"{paper['pdf_sha256']}.middle.json.zst").exists():continue
            if wanted is None or paper['forum_id'] in wanted:jobs.append((str(root),paper,str(output),str(args.figure_regions) if args.figure_regions else None))
    if wanted is not None and wanted!={j[1]['forum_id'] for j in jobs}:raise ValueError('Missing sample IDs')
    identities=[(j[0],j[1]['forum_id']) for j in jobs]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate source record identities')
    provenance={'format':VERSION+('+mineru-figures' if args.figure_regions else ''),'figure_regions':str(args.figure_regions) if args.figure_regions else None,'code_sha256':code_hash,'inputs':inputs,'rows':len(jobs),
                'sample_ids':sorted(wanted) if wanted is not None else None}
    output.mkdir(parents=True,exist_ok=True)
    lockfile=(output/'build.lock').open('a')
    import fcntl
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'provenance.json').exists():
        if json.loads((output/'provenance.json').read_text())!=provenance:raise ValueError('Output belongs to another cleanup revision/input set')
    else:atomic(output/'provenance.json',json.dumps(provenance,indent=2).encode())
    import pyarrow as pa
    import pyarrow.parquet as pq
    started=time.time();shards=[];counts={};flagged=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0,len(jobs),250):
            batch=jobs[offset:offset+250];path=output/'data'/f'train-{offset//250:05}.parquet'
            if path.exists():
                rows=pq.read_table(path).to_pylist()
                if len(rows)!=len(batch):raise ValueError('Incomplete existing shard')
                for row,job in zip(rows,batch):
                    paper=job[1]
                    if row['id']!=paper['forum_id'] or row['source_text_sha256']!=paper['text_sha256']:raise ValueError('Existing shard identity mismatch')
                    raw=(output/row['clean_positions_path']).read_bytes()
                    if digest(raw)!=row['clean_blob_sha256']:raise ValueError('Existing clean sidecar mismatch')
                    result=decode(raw)
                    if result['text']!=row['text'] or digest(row['text'].encode())!=row['text_sha256']:raise ValueError('Existing shard text mismatch')
            else:
                rows=list(pool.map(one,batch))
                path.parent.mkdir(exist_ok=True)
                temp=path.with_suffix('.tmp');pq.write_table(pa.Table.from_pylist(rows),temp,compression='zstd')
                if pq.read_table(temp).to_pylist()!=rows:raise ValueError('Parquet readback mismatch')
                temp.replace(path)
            for row in rows:
                for k,v in json.loads(row['stats_json'])['rules'].items():counts[k]=counts.get(k,0)+v
                flagged+=row['flag_count']>0
            shards.append({'path':str(path.relative_to(output)),'rows':len(rows),'sha256':digest(path.read_bytes())})
            status={'state':'building','rows':offset+len(rows),'total':len(jobs),'rule_counts':counts,
                    'papers_with_flags':flagged,'elapsed_seconds':time.time()-started}
            atomic(output/'status.json',json.dumps(status).encode());print(json.dumps(status),flush=True)
    # Re-read every published shard at completion; no completion marker on a partial run.
    for s in shards:
        if digest((output/s['path']).read_bytes())!=s['sha256']:raise ValueError('Shard changed during build')
    done={**provenance,'state':'verified','shards':shards,'rule_counts':counts,
          'papers_with_flags':flagged,'elapsed_seconds':time.time()-started,
          'source_modified':False,'model_inference_runs':0}
    atomic(output/'completion.json',json.dumps(done,indent=2).encode())
    atomic(output/'status.json',json.dumps(done).encode())
    print(json.dumps({k:v for k,v in done.items() if k not in ['shards','sample_ids']}),flush=True)


if __name__=='__main__':main()
