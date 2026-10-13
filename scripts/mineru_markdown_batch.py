"""PDF -> Markdown with MinerU2.5-Pro (VLM), for a reference/structure dataset.

Run on the training Space. One vLLM server (BF16) is started on the GPU given by
--gpu; several CPU client processes feed it concurrently through MinerU's local
parser API (never the mineru.net remote service). Output is model-generated
Markdown and is kept separate from the canonical text-layer extractions: it must
not be used as human-authored training text. Resumable via receipts.jsonl.
"""
import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import urllib.request


def atomic(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def papers(source):
    import pyarrow.parquet as pq
    rows=[]
    for shard in sorted((source/'data').glob('train-*.parquet')):
        rows+=pq.read_table(shard,columns=['id','pdf_sha256','pdf_path']).to_pylist()
    unique={}
    for r in rows:unique.setdefault(r['pdf_sha256'],r)   # identical PDFs are parsed once
    ordered=sorted(unique.values(),key=lambda r:r['pdf_sha256'])
    random.Random(0).shuffle(ordered)   # partial runs are a uniform sample
    return ordered


def worker(args):
    """One client process: parse its share of PDFs with bounded async concurrency."""
    source,output,server,concurrency,jobs,gpu=args
    # MinerU's small layout/OCR models run in the client: keep them on the job's GPU,
    # never on GPUs other jobs are using.
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    os.environ['HF_HUB_OFFLINE']='1'
    import zstandard
    from mineru.config import VlmConfig
    from mineru.parser import parse_async
    config=VlmConfig(engine='vllm',server_url=server,max_concurrency=64,http_timeout=1800)
    gate=asyncio.Semaphore(concurrency);results=[]
    async def one(row):
        async with gate:
            started=time.time();pdf=Path(source)/row['pdf_path']
            try:
                raw=pdf.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=row['pdf_sha256']:raise ValueError('PDF checksum mismatch')
                result=await parse_async(pdf,tier='standard',image_analysis=False,vlm_config=config)
                # No embedded base64 figures (~2 MB/paper); the layout JSON keeps the regions.
                markdown=result.markdown(image_renderer=lambda block:'').encode()
                middle=zstandard.ZstdCompressor(level=10).compress(result.to_json().encode())
                base=Path(output)/'objects'/row['pdf_sha256']
                atomic(base.with_suffix('.md'),markdown);atomic(base.with_suffix('.middle.json.zst'),middle)
                receipt={'pdf_sha256':row['pdf_sha256'],'id':row['id'],'state':'parsed','pages':len(result.pages),
                         'markdown_sha256':hashlib.sha256(markdown).hexdigest(),'markdown_bytes':len(markdown),
                         'middle_sha256':hashlib.sha256(middle).hexdigest(),'seconds':time.time()-started}
            except Exception as exc:
                receipt={'pdf_sha256':row['pdf_sha256'],'id':row['id'],'state':'failed',
                         'error':f'{type(exc).__name__}: {str(exc)[:500]}','seconds':time.time()-started}
            with open(Path(output)/f'receipts-{os.getpid()}.jsonl','a') as log:
                log.write(json.dumps(receipt)+'\n')
            results.append(receipt)
    async def main():await asyncio.gather(*(one(r) for r in jobs))
    asyncio.run(main())
    return results


def wait_ready(server,process,timeout=1800):
    end=time.time()+timeout
    while time.time()<end:
        if process.poll() is not None:raise RuntimeError('vLLM server exited during startup')
        try:
            with urllib.request.urlopen(server+'/v1/models',timeout=5) as r:
                if r.status==200:return
        except Exception:pass
        time.sleep(5)
    raise TimeoutError('vLLM server did not become ready')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path,help='Dataset with data/train-*.parquet (id, pdf_sha256, pdf_path)')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gpu',required=True)
    p.add_argument('--clients',type=int,default=12)
    p.add_argument('--concurrency',type=int,default=4,help='Concurrent documents per client process')
    p.add_argument('--gpu-memory-utilization',default='0.45')
    p.add_argument('--port',type=int,default=30000)
    p.add_argument('--limit',type=int,default=0,help='Process only the first N pending PDFs (calibration)')
    args=p.parse_args()
    output=args.output.resolve();source=args.source.resolve();output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    done=set()
    for receipts in output.glob('receipts-*.jsonl'):
        for line in receipts.read_text().splitlines():
            r=json.loads(line)
            if r['state']=='parsed':done.add(r['pdf_sha256'])
    todo=[r for r in papers(source) if r['pdf_sha256'] not in done]
    if args.limit:todo=todo[:args.limit]
    bindir=Path(sys.executable).parent
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu   # inherited by client workers too
    # MinerU spawns its own subprocesses; with default math-library thread pools
    # (~47 threads each) the container hits its shared thread limit. Keep them small.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    # The Space has no CUDA toolkit (nvcc): use vLLM's prebuilt FlashAttention
    # kernels and sampler instead of FlashInfer's JIT-compiled ones.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HUB_OFFLINE='1',
             VLLM_ATTENTION_BACKEND='FLASH_ATTN',VLLM_USE_FLASHINFER_SAMPLER='0')
    server_cmd=[str(bindir/'mineru-openai-server'),'--engine','vllm','--host','127.0.0.1','--port',str(args.port),
                '--dtype','bfloat16','--gpu-memory-utilization',args.gpu_memory_utilization]
    provenance={'source_dataset':source.name,'gpu':args.gpu,'server_command':server_cmd[1:],'tier':'standard',
                'image_analysis':False,'model':'MinerU2.5-Pro-2605-1.2B','inference_dtype':'bfloat16',
                'pending':len(todo),'already_done':len(done),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'notice':'Model-generated Markdown; not human-authored text. Keep separate from training text.'}
    atomic(output/'provenance.json',json.dumps(provenance,indent=2).encode())
    print(json.dumps({'pending':len(todo),'already_done':len(done)}),flush=True)
    server=f'http://127.0.0.1:{args.port}'
    log=open(output/'vllm-server.log','a')
    vllm=subprocess.Popen(server_cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(output/'server-process.json',json.dumps({'pid':vllm.pid,'gpu':args.gpu}).encode())
    try:
        wait_ready(server,vllm)
        started=time.time();completed=0;failed=0;pages=0
        # Hand out work in waves so progress is recorded and visible throughout.
        wave=args.clients*args.concurrency*8
        with ProcessPoolExecutor(max_workers=args.clients) as pool:
            for offset in range(0,len(todo),wave):
                batch=todo[offset:offset+wave];shares=[batch[i::args.clients] for i in range(args.clients)]
                wave_failed=0
                for results in pool.map(worker,[(str(source),str(output),server,args.concurrency,s,args.gpu) for s in shares if s]):
                    for r in results:
                        completed+=1;failed+=r['state']!='parsed';wave_failed+=r['state']!='parsed';pages+=r.get('pages',0)
                if wave_failed>.2*len(batch):
                    # Systematic failure (e.g. GPU out of memory): stop instead of burning
                    # through the queue; failed PDFs stay pending for the next run.
                    atomic(output/'status.json',json.dumps({'state':'aborted_high_failure_rate','completed':completed,
                           'failed':failed,'wave_failed':wave_failed,'wave_size':len(batch)}).encode())
                    raise SystemExit(f'Aborting: {wave_failed}/{len(batch)} failures in one wave')
                elapsed=time.time()-started
                status={'state':'running','completed':completed,'failed':failed,'pending':len(todo),'pages':pages,
                        'pages_per_second':pages/max(1,elapsed),'elapsed_seconds':elapsed,
                        'eta_hours':(len(todo)-completed)*(elapsed/max(1,completed))/3600}
                atomic(output/'status.json',json.dumps(status).encode());print(json.dumps(status),flush=True)
        atomic(output/'status.json',json.dumps({**status,'state':'complete' if not args.limit else 'calibrated'}).encode())
    finally:
        try:os.killpg(vllm.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:vllm.wait(timeout=120)
        except subprocess.TimeoutExpired:os.killpg(vllm.pid,signal.SIGKILL)


if __name__=='__main__':main()
