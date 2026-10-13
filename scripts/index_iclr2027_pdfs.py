"""Publish public OpenReview discovery metadata; no inference or PDF changes."""
import json, hashlib, time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from upload_paper_pdfs import BASE, DATA, request


def public_value(note, key):
    if 'everyone' not in note.get('readers', []):
        return None
    field = note.get('content', {}).get(key)
    if isinstance(field, dict):
        if 'readers' in field and 'everyone' not in field['readers']:
            return None
        return field.get('value')
    return field


def build_index(root):
    metadata = {p['id']: p for p in map(json.loads, (root/'manifest.jsonl').open())}
    notes = {}
    files = sorted((root/'enumeration-pages').glob('*.json')) + sorted((root/'enumeration-runs').glob('*/*.json'))
    for file in files:
        for note in json.loads(file.read_text()).get('notes', []):
            old = notes.get(note['id'])
            if old is None or note.get('mdate', 0) >= old.get('mdate', 0):
                notes[note['id']] = note
    rows, shards = {}, {}
    for file in sorted((root/'chunks').glob('*/manifest.json')):
        for paper in json.loads(file.read_text()).get('papers', []):
            digest = paper.get('sha256', '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                continue
            forum = paper['id']; note = notes.get(forum, {})
            if 'everyone' not in note.get('readers', []):
                continue
            def text(key):
                value = public_value(note, key)
                return value if isinstance(value, str) else ''
            keywords = public_value(note, 'keywords') or []
            if isinstance(keywords, str): keywords = [keywords]
            keywords = [k for k in keywords if isinstance(k, str)]
            detail = dict(abstract=text('abstract'), bibtex=text('_bibtex'), license=note.get('license', ''), venue=text('venue'))
            row = dict(key=f'papers/{digest}.pdf', title=text('title') or metadata[forum]['title'], filename=forum+'.pdf', conference='iclr', year=2027,
                       forum_id=note.get('forum', forum), number=note.get('number'), keywords=keywords, primary_area=text('primary_area'),
                       tldr=text('TLDR'), abstract_preview=text('abstract')[:420], published=note.get('odate'), updated=note.get('mdate'))
            rows[row['key']] = row
            shards.setdefault(digest[:2], {})[digest[:24]] = detail
    return {'papers': list(rows.values())}, shards


def main():
    index, shards = build_index(DATA/'openreview_iclr2027_all')
    receipts=DATA/'openreview_iclr2027_all'/'metadata-publication-receipts'
    receipts.mkdir(exist_ok=True)
    def publish_json(key, value):
        raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(raw).hexdigest();proof=receipts/(hashlib.sha256(key.encode()).hexdigest()+'.json')
        if proof.exists():
            saved=json.loads(proof.read_text())
            if saved.get('sha256')==digest and saved.get('readback_verified'):return
        url=BASE+'/'+key
        # Compare current contents before PUT, so independent publishers' identical writes are reused.
        try:unchanged=request(url)==value
        except Exception:unchanged=False
        if not unchanged:request(url,raw,'application/json')
        assert request(url)==value
        temp=proof.with_suffix('.tmp');temp.write_text(json.dumps({'key':key,'sha256':digest,'readback_verified':True,'at':time.time()}));temp.replace(proof)
    def publish_shard(pair):
        prefix, details = pair
        publish_json(f'indexes/iclr2027-details/{prefix}.json',details)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(publish_shard, sorted(shards.items())))
    discovery = {'papers': [dict(id=row['key'][7:31], **{k:v for k,v in row.items() if k not in ('key','conference','year')}) for row in index['papers']]}
    publish_json('indexes/iclr2027-discovery.json',discovery)
    index = {'papers': [{k:v for k,v in row.items() if k in ('key','title','filename','conference','year','forum_id')} for row in index['papers']]}
    publish_json('indexes/iclr2027-pdfs.json',index)
    from publish_browse_snapshot import main as publish_browse
    publish_browse()
    print(f'Verified public metadata for {len(index["papers"])} PDFs and {len(shards)} detail shards.')


if __name__ == '__main__':
    main()
