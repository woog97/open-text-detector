"""Prepare immutable, PDF-verified browsing pages during upload publication."""
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path
from upload_paper_pdfs import ROOT, BASE, request, remote_objects


# Models whose summaries carry `flagged` (sentences above the 1%-false-positive threshold), in the order the site
# prefers them (paper-scores.ts DEFAULT_MODELS). Uncalibrated summaries give no flagged share.
FLAG_MODELS = ('qwen36-35b-a3b',)


def add_flags(rows, published, override=None):
    """Put each paper's flagged counts on its browse row as `flags` {mode: [flagged, counted sentences]}.

    `override` (indexes/flags/current.json, when published) holds recounted counts per counting mode (bibliography and
    appendices in or out, each with its own recalibrated 1%-FPR threshold): {model, thresholds, papers: {id: {modes,
    sections}}}. Papers it does not list fall back to the catalogue summary, which counts every sentence ('all')."""
    summaries = {p['id']: p.get('score_summaries') or {} for p in published['items']}
    papers = (override or {}).get('papers', {})
    n = 0
    for row in rows:
        for k in ('flagged', 'sentences', 'flag_model', 'bib_sentences', 'flags', 'sections', 'mean_score'):
            row.pop(k, None)
        entry = papers.get(row['id'])
        if entry:
            row.update(flags=entry['modes'], sections=entry['sections'], flag_model=override['model'])
            if entry.get('mean'): row['mean_score'] = entry['mean']  # average sentence score per base mode (Oct 13, 2026)
            # Fields read by Atlas builds deployed before counting modes (Oct 8, 2026): the bibliography-excluded counts.
            row.update(flagged=entry['modes']['nobib'][0], sentences=entry['modes']['nobib'][1], bib_sentences=entry['sections'][1]); n += 1; continue
        s = summaries.get(row['id'], {})
        model = next((m for m in FLAG_MODELS if 'flagged' in (s.get(m) or {})), None)
        if model:
            row.update(flags={'all': [s[model]['flagged'], s[model]['total']]}, flag_model=model); n += 1
    return n


def flag_override():
    try:
        return request(BASE+'/indexes/flags/current.json')
    except RuntimeError:
        return None


def build_rows(objects, published, archive, discovery, enrichment=None):
    public = {p['pdf_key']: p for p in published['items']}
    metadata = {p['key']: p for p in archive['papers']}
    rich = {p['id']: p for p in (enrichment or {}).get('papers', [])}
    rich.update({p['id']: {**p, 'collection':'iclr/2027'} for p in discovery['papers']})
    rows = []
    for key, obj in sorted(objects.items()):
        digest = key.removeprefix('papers/').removesuffix('.pdf')
        if key != f'papers/{digest}.pdf' or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            continue
        p = public.get(key, {}); meta = metadata.get(key, {})
        custom = obj.get('custom_metadata') or {}
        extra = rich.get(digest[:24], {})
        filename = p.get('filename') or meta.get('filename') or custom.get('filename') or digest+'.pdf'
        collection = p.get('collection') or meta.get('collection') or (f"{meta['conference']}/{meta['year']}" if meta.get('conference') and meta.get('year') else custom.get('collection','Other uploads'))
        if extra.get('collection'): collection = extra['collection']
        row = dict(id=p.get('id',digest[:24]),title=p.get('title') or meta.get('title') or custom.get('title') or filename.removesuffix('.pdf'),filename=filename,collection=collection,bytes=obj['size'],classified=bool(p.get('classified')),models=p.get('models',[]))
        row.update({k:v for k,v in extra.items() if k not in ('id','key','bytes','classified','models','collection')})
        rows.append(row)
    add_flags(rows, published, flag_override())
    assert len({p['id'] for p in rows}) == len(rows), 'Duplicate public IDs'
    return rows


def main(prepared=False, enrich=False, flags=False):
    out = ROOT/'app/.sites-runtime/browse';out.mkdir(parents=True,exist_ok=True)
    with (out/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source_revision=None
        if flags:
            # Refresh only the flagged-share fields of the live snapshot from the live catalogue.
            source=request(BASE+'/indexes/browse/current.json');source_revision=source['revision']
            rows=request(BASE+'/'+source['index_key'])['items']
            print(f'flagged shares on {add_flags(rows,request(BASE+"/atlas-public/catalogue.json"),flag_override()):,} of {len(rows):,} papers',flush=True)
        elif enrich:
            source=request(BASE+'/indexes/browse/current.json');source_revision=source['revision']
            rows=request(BASE+'/'+source['index_key'])['items']
            metadata={p['id']:p for p in request(BASE+'/indexes/paper-discovery.json')['papers']}
            canonical={p['id']:p['pdf_key'][7:31] for p in request(BASE+'/atlas-public/catalogue.json')['items']}
            for row in rows:
                extra=metadata.get(canonical.get(row['id'],row['id']),{})
                row.update({k:v for k,v in extra.items() if k not in ('id','key','bytes','classified','models','collection','flagged','sentences','flag_model','bib_sentences','flags','sections','mean_score')})
        elif prepared:
            receipt=json.loads((out/'receipt.json').read_text())
            assert request(BASE+'/indexes/browse/current.json')['revision']==receipt['revision'], 'Newer snapshot published; rebuild from live inventory'
            rows=json.loads((out/'papers.json').read_text())['items']
        else:
            objects=remote_objects()
            rows=build_rows(objects,request(BASE+'/atlas-public/catalogue.json'),request(BASE+'/indexes/downloaded-papers.json'),request(BASE+'/indexes/iclr2027-discovery.json'),request(BASE+'/indexes/paper-discovery.json') if 'indexes/paper-discovery.json' in objects else None)
        raw=json.dumps({'items':rows},ensure_ascii=False,separators=(',',':')).encode()
        (out/'papers.json').write_bytes(raw)
        # Home snapshot, compact search index and card pack, from the same code the live endpoint runs.
        code="""import fs from 'node:fs';import {browsePage} from './app/lib/browse-page.ts';import {buildIndex,indexJson} from './app/lib/browse-index.ts';const [src,dir]=process.argv.slice(1);const items=JSON.parse(fs.readFileSync(src,'utf8')).items;fs.writeFileSync(dir+'/home.json',JSON.stringify(browsePage(items,new URLSearchParams())));const {index,pack}=buildIndex(items);fs.writeFileSync(dir+'/search.json',indexJson(index));fs.writeFileSync(dir+'/cards.jsonl',pack);"""
        subprocess.run(['node','--experimental-strip-types','--input-type=module','-e',code,str(out/'papers.json'),str(out)],cwd=ROOT,check=True)
        home=json.loads((out/'home.json').read_text())
        revision=hashlib.sha256(raw+json.dumps(home,sort_keys=True).encode()).hexdigest()
        home['revision']=revision
        base=f'indexes/browse/{revision}'
        manifest=dict(revision=revision,index_key=f'{base}/papers.json',home_key=f'{base}/home.json',search_key=f'{base}/search.json',cards_key=f'{base}/cards.jsonl',published_at=time.time(),total=len(rows))
        def put(key,data,content_type='application/json'):
            request(BASE+'/'+key,data,content_type)
            assert hashlib.sha256(request(BASE+'/'+key,raw=True)).digest()==hashlib.sha256(data).digest(), f'Readback mismatch: {key}'
        put(manifest['index_key'],raw)
        put(manifest['home_key'],json.dumps(home,ensure_ascii=False,separators=(',',':')).encode())
        put(manifest['search_key'],(out/'search.json').read_bytes())
        put(manifest['cards_key'],(out/'cards.jsonl').read_bytes(),'application/x-ndjson')
        # Advance only after every immutable object passed readback verification.
        if source_revision:assert request(BASE+'/indexes/browse/current.json')['revision']==source_revision, 'Newer snapshot published; retry'
        put('indexes/browse/current.json',json.dumps(manifest).encode())
        (out/'receipt.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Published fast browse snapshot: {len(rows)} papers; first page {len(json.dumps(home).encode()):,} bytes; revision {revision}',flush=True)


if __name__=='__main__':
    import sys
    main(prepared='--prepared' in sys.argv, flags='--flags' in sys.argv)
