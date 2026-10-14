"""List the reader feedback messages ("Think this result is wrong?") stored by the public Worker (app/atlas-cloud/feedback.ts).

Messages are private R2 objects under feedback/YYYY-MM-DD/; nothing on the site reads them back. This script lists them
through the Cloudflare API with your wrangler login (the same access as the publish scripts) and prints them newest first,
with the paper's title and a reader link. Nothing is changed or deleted.
Includes site-wide messages from the corner Feedback button (kind "site", stored under feedback/site/).
Usage: uv run -q --python 3.12 scripts/read_feedback.py [--since YYYY-MM-DD] [--json]
"""
import gzip, json, re, sys, urllib.parse, urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from upload_paper_pdfs import BASE, request

PREFIX = 'feedback/'
SITE = 'https://opentextdetector.com'


def marked_sentences(m):
    """Text of the passages the visitor marked, from the same report they were viewing (public reader endpoint)."""
    if not m.get('passages'): return []
    req = urllib.request.Request(f"{SITE}/backend/v1/pdf-reader/{m['paper']}", headers={'Accept-Encoding': 'gzip', 'User-Agent': 'Mozilla/5.0 (compatible; otd-read-feedback/1.0; +https://github.com/woog97/open-text-detector)'})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read(); raw = gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw
    reports = json.loads(raw).get('reports', [])
    rep = next((x for x in reports if x['id'] == m.get('context', {}).get('report')), None)
    if not rep: return [(i, '(report no longer published; passage text unavailable)') for i in m['passages']]
    segs, text = rep['result']['segments'], rep['text']
    return [(i, f"[score {segs[i]['score']:.4f}] " + text[segs[i]['start']:segs[i]['end']].strip()) if i < len(segs) else (i, '(out of range)') for i in m['passages']]


def keys(since=''):
    found, cursor = [], None
    while True:
        params = {'per_page': 1000, 'prefix': PREFIX}
        if cursor: params['cursor'] = cursor
        page = request(BASE + '?' + urllib.parse.urlencode(params))
        # Paper messages: feedback/YYYY-MM-DD/…; site-wide messages (corner button): feedback/site/YYYY-MM-DD/….
        found += [i['key'] for i in page['result'] if re.search(r'/(\d{4}-\d{2}-\d{2})/', i['key']) and re.search(r'/(\d{4}-\d{2}-\d{2})/', i['key']).group(1) >= since]
        info = page.get('result_info', {})
        if not info.get('is_truncated'): return sorted(found, reverse=True)
        cursor = info['cursor']


def main(argv):
    since = argv[argv.index('--since') + 1] if '--since' in argv else ''
    messages = sorted((json.loads(request(BASE + '/' + urllib.parse.quote(k, safe=''), raw=True)) for k in keys(since)), key=lambda m: m['received_at'], reverse=True)
    if '--json' in argv: print(json.dumps(messages, indent=1, ensure_ascii=False)); return
    titles = {p['id']: p.get('title', '') for p in json.loads(request(BASE + '/atlas-public%2Fcatalogue.json', raw=True))['items']} if messages else {}
    for m in messages:
        when = datetime.fromisoformat(m['received_at'].replace('Z', '+00:00')).astimezone(ZoneInfo('America/Los_Angeles'))
        ctx = ', '.join(f'{k} {v}' for k, v in m.get('context', {}).items())
        if m.get('kind') == 'site': print(f"{when:%Y-%m-%d %H:%M %Z}  Site feedback ({ctx or 'page unknown'})\n")
        else: print(f"{when:%Y-%m-%d %H:%M %Z}  {titles.get(m['paper'], m['paper'])}\n  https://opentextdetector.com/?paper={m['paper']}  ({ctx})\n")
        if m['comment']: print('  ' + m['comment'].replace('\n', '\n  ') + '\n')
        marks = marked_sentences(m)
        if marks:
            print(f'  Marked as written without AI ({len(marks)}):')
            for i, t in marks: print(f'    #{i + 1}  {t[:300]}')
            print()
    print(f'{len(messages)} message{"" if len(messages) == 1 else "s"}' + (f' since {since}' if since else ''))


if __name__ == '__main__':
    main(sys.argv[1:])
