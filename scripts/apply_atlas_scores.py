"""Publish prepared score summaries only when the live catalogue still matches."""
import fcntl
import json
import gzip, hashlib, tomllib, urllib.request
from copy import deepcopy
from atlas_scores import summarize
from pathlib import Path
from upload_paper_pdfs import BASE, CONFIG, request

OUT = Path(__file__).resolve().parents[1] / 'research/exports/atlas-public'

def main():
    with (OUT / 'publication.lock').open('a') as lock:
        # Never race an active classification publisher or its watcher.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        patch = json.loads((OUT / 'score-patch.json').read_text())
        catalogue = request(BASE + '/atlas-public/catalogue.json')
        original = deepcopy(catalogue)
        ledger = json.loads((OUT / 'ledger.json').read_text())
        for item in catalogue['items']:
            p = patch[item['id']]
            if item['detail_key'] == p['detail_key']:
                item['score_summaries'] = p['score_summaries']
            else:
                token = tomllib.loads(CONFIG.read_text())['oauth_token']
                req = urllib.request.Request(BASE + '/' + item['detail_key'], headers={'Authorization':'Bearer '+token})
                with urllib.request.urlopen(req, timeout=120) as response:
                    blob = response.read()
                assert hashlib.sha256(blob).hexdigest() == Path(item['detail_key']).name.split('.')[0]
                detail = json.loads(gzip.decompress(blob))
                item['score_summaries'] = {r['model']['name'].split()[-1]: summarize(r['result']['segments'],r['text']) for r in detail['reports']}
                assert set(item['score_summaries']) == set(item['models'])
            saved = ledger[p['pdf_sha256']]
            assert saved['fingerprint'] == p['fingerprint'], 'Local results changed; prepare scores again'
            saved['item']['score_summaries'] = p['score_summaries']
        assert request(BASE + '/atlas-public/catalogue.json') == original, 'Live catalogue changed during preparation'
        raw = json.dumps(catalogue, separators=(',', ':')).encode()
        request(BASE + '/atlas-public/catalogue.json', raw, 'application/json')
        assert request(BASE + '/atlas-public/catalogue.json') == catalogue, 'Remote verification failed'
        for name, value in [('catalogue.json', catalogue), ('ledger.json', ledger)]:
            temp = OUT / (name + '.scores.tmp')
            temp.write_text(json.dumps(value, separators=(',', ':')))
            temp.replace(OUT / name)
        print(f"Verified scores published for {len(catalogue['items']):,} papers")

if __name__ == '__main__':
    main()
