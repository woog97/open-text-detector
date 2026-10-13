"""Atlas catalogue, flag index and browse snapshot for the Oct 10 model (runs on the Mac).

stage   BASE_CATALOGUE.json INDEX.parquet FLAGS_OCT10.json LIVE_BROWSE_PAPERS.json RECAL.json STAGE_DIR
        Builds STAGE_DIR/catalogue.json from BASE (the live atlas-public/catalogue.json, saved locally): every item whose
        PDF has an Oct 10 object (build_oct10.py index.parquet) gets models = ['qwen36-35b-a3b-oct10'],
        score_summaries = {'qwen36-35b-a3b-oct10': summary} (the Oct 7 and 4B summaries are dropped) and detail_key = the
        Oct 10 object. Other items (MELD-scored collections) are unchanged. The catalogue's model list drops qwen35-4b and
        qwen36-35b-a3b and gains the Oct 10 entry with its calibration. Copies FLAGS_OCT10 to STAGE_DIR/flags.json, and
        builds the browse snapshot (papers.json, home.json, search.json, cards.jsonl) the way
        scripts/publish_browse_snapshot.py --flags would after publication (live rows, flag fields from the staged
        files), then loads search.json through app/lib/browse-index.ts (validateIndex) as a check. Writes
        STAGE_DIR/stage.json with the source hashes. Nothing is uploaded.
publish STAGE_DIR [--dry-run]
        Uploads the staged catalogue and flag index to R2 through the Cloudflare API (scripts/upload_paper_pdfs.request),
        but only when the live catalogue and flag index are still byte-identical to the ones the stage was built from.
        Saves the live copies to STAGE_DIR/rollback/ first, then reads every upload back. Afterwards run
        scripts/publish_public_catalogue.py (reader lookup shards + public catalogue) and
        scripts/publish_browse_snapshot.py --flags (browse snapshot).
rollback STAGE_DIR
        Puts STAGE_DIR/rollback/{catalogue.json,flags.json} back (then rerun the two scripts above).
Detail objects must be uploaded (upload_oct10.py) before `publish`.
"""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
MODEL = 'qwen36-35b-a3b-oct10'
OLD = ('qwen35-4b', 'qwen36-35b-a3b')
KEYS = {'catalogue.json': 'atlas-public/catalogue.json', 'flags.json': 'indexes/flags/current.json'}
sha = lambda b: hashlib.sha256(b).hexdigest()


def stage(base, index, flags, browse, recal, out):
    import pyarrow.parquet as pq
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True); raw = Path(base).read_bytes(); cat = json.loads(raw)
    rows = {r['pdf_sha256']: r for r in pq.read_table(index).to_pylist() if not r.get('error')}
    fl = json.loads(Path(flags).read_text()); th = fl['thresholds']; items = []; n = 0
    one = next(t for t in json.loads(Path(recal).read_text())['modes']['all']['thresholds'] if t['target_fpr'] == 0.01)  # summary threshold
    assert one['threshold'] == th['all']
    for p in cat['items']:
        r = rows.get(p['pdf_key'][7:-4])
        if r is None: items.append(p); continue
        assert r['id'] == p['id'], (r['id'], p['id'])
        items.append({**p, 'models': [MODEL], 'score_summaries': {MODEL: json.loads(r['summary_json'])}, 'detail_key': r['detail_key']}); n += 1
    entry = {'id': MODEL, 'name': 'Qwen3.6 35B-A3B MoE · Oct 10', 'available': n, 'total': len(items), 'complete': False,
             'calibration': {'sentence_threshold_1pct_fpr': one['threshold'], 'heldout_fpr': one['heldout_fpr'], 'human_papers': 2400, 'page': '/?view=calibration',
                             'flag_thresholds': th, 'hidden_modes': fl['hidden_modes']}}
    new = {**cat, 'published_at': time.time(), 'items': items, 'models': [m for m in cat['models'] if m['id'] not in OLD + (MODEL,)] + [entry]}
    body = json.dumps(new, separators=(',', ':')).encode(); (out / 'catalogue.json').write_bytes(body)
    shutil.copyfile(flags, out / 'flags.json')
    # Browse snapshot, as publish_browse_snapshot.py --flags builds it.
    from publish_browse_snapshot import add_flags
    brows = json.loads(Path(browse).read_text())['items']; got = add_flags(brows, new, fl)
    braw = json.dumps({'items': brows}, ensure_ascii=False, separators=(',', ':')).encode(); bdir = out / 'browse'; bdir.mkdir(exist_ok=True)
    (bdir / 'papers.json').write_bytes(braw)
    code = """import fs from 'node:fs';import {browsePage} from './app/lib/browse-page.ts';import {buildIndex,indexJson} from './app/lib/browse-index.ts';const [src,dir]=process.argv.slice(1);const items=JSON.parse(fs.readFileSync(src,'utf8')).items;fs.writeFileSync(dir+'/home.json',JSON.stringify(browsePage(items,new URLSearchParams())));const {index,pack}=buildIndex(items);fs.writeFileSync(dir+'/search.json',indexJson(index));fs.writeFileSync(dir+'/cards.jsonl',pack);"""
    subprocess.run(['node', '--experimental-strip-types', '--input-type=module', '-e', code, str(bdir / 'papers.json'), str(bdir)], cwd=ROOT, check=True)
    check = """import fs from 'node:fs';import {BrowseIndex} from './app/lib/browse-index.ts';const x=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));const b=new BrowseIndex(x);const r=b.query({q:'',collection:'iclr/2027',area:'',tag:'',saved:false,sort:'flagged-asc',ai:'',mode:'nobib0.1'},new Set(),null,1);console.log(JSON.stringify({papers:b.n,iclr_total:r.total,histogram:r.histogram}));"""
    res = subprocess.run(['node', '--experimental-strip-types', '--input-type=module', '-e', check, str(bdir / 'search.json')], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    info = {'base_catalogue_sha256': sha(raw), 'live_flags_sha256': None, 'catalogue_sha256': sha(body), 'items': len(items), 'oct10_items': n,
            'browse_rows': len(brows), 'browse_rows_with_flags': got, 'browse_check': json.loads(res), 'staged_at': time.time()}
    lf = Path(base).with_name('flags-current.json')
    if lf.exists(): info['live_flags_sha256'] = sha(lf.read_bytes())
    (out / 'stage.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info))


def publish(out, dry):
    from upload_paper_pdfs import BASE, request
    out = Path(out); info = json.loads((out / 'stage.json').read_text()); rb = out / 'rollback'; rb.mkdir(exist_ok=True)
    live = {f: request(BASE + '/' + k, raw=True) for f, k in KEYS.items()}
    stamp = time.strftime('%Y%m%dT%H%M%S')
    for f, b in live.items():
        (rb / f'{stamp}-{f}').write_bytes(b)
        if not (rb / f).exists(): (rb / f).write_bytes(b)  # first pre-publication copy is the rollback target
    if sha(live['catalogue.json']) != info['base_catalogue_sha256']: raise SystemExit('live catalogue changed since staging; restage')
    if info['live_flags_sha256'] and sha(live['flags.json']) != info['live_flags_sha256']: raise SystemExit('live flag index changed since staging; restage')
    print('live copies saved to', rb, '; dry run' if dry else '', flush=True)
    if dry: return
    for f, k in KEYS.items():
        body = (out / f).read_bytes(); request(BASE + '/' + k, body, 'application/json')
        assert sha(request(BASE + '/' + k, raw=True)) == sha(body), f'readback mismatch {k}'
        print('published', k, len(body), flush=True)


def rollback(out):
    from upload_paper_pdfs import BASE, request
    rb = Path(out) / 'rollback'
    for f, k in KEYS.items():
        body = (rb / f).read_bytes(); request(BASE + '/' + k, body, 'application/json')
        assert sha(request(BASE + '/' + k, raw=True)) == sha(body); print('restored', k, flush=True)


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'stage': stage(*a)
    elif cmd == 'publish': publish(a[0], '--dry-run' in a)
    else: rollback(a[0])
