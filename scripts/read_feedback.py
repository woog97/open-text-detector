"""List the reader feedback messages ("Think this result is wrong?") stored by the public Worker (app/atlas-cloud/feedback.ts).

Messages are private R2 objects under feedback/YYYY-MM-DD/; nothing on the site reads them back. This script lists them
through the Cloudflare API with your wrangler login (the same access as the publish scripts) and prints them newest first,
with the paper's title and a reader link. Nothing is changed or deleted.
Usage: uv run -q --python 3.12 scripts/read_feedback.py [--since YYYY-MM-DD] [--json]
"""
import json, sys, urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo
from upload_paper_pdfs import BASE, request

PREFIX = 'feedback/'


def keys(since=''):
    found, cursor = [], None
    while True:
        params = {'per_page': 1000, 'prefix': PREFIX}
        if cursor: params['cursor'] = cursor
        page = request(BASE + '?' + urllib.parse.urlencode(params))
        found += [i['key'] for i in page['result'] if i['key'][len(PREFIX):len(PREFIX) + 10] >= since]
        info = page.get('result_info', {})
        if not info.get('is_truncated'): return sorted(found, reverse=True)
        cursor = info['cursor']


def main(argv):
    since = argv[argv.index('--since') + 1] if '--since' in argv else ''
    messages = [json.loads(request(BASE + '/' + urllib.parse.quote(k, safe=''), raw=True)) for k in keys(since)]
    if '--json' in argv: print(json.dumps(messages, indent=1, ensure_ascii=False)); return
    titles = {p['id']: p.get('title', '') for p in json.loads(request(BASE + '/atlas-public%2Fcatalogue.json', raw=True))['items']} if messages else {}
    for m in messages:
        when = datetime.fromisoformat(m['received_at'].replace('Z', '+00:00')).astimezone(ZoneInfo('America/Los_Angeles'))
        ctx = ', '.join(f'{k} {v}' for k, v in m.get('context', {}).items())
        print(f"{when:%Y-%m-%d %H:%M %Z}  {titles.get(m['paper'], m['paper'])}\n  https://opentextdetector.com/?paper={m['paper']}  ({ctx})\n")
        print('  ' + m['comment'].replace('\n', '\n  ') + '\n')
    print(f'{len(messages)} message{"" if len(messages) == 1 else "s"}' + (f' since {since}' if since else ''))


if __name__ == '__main__':
    main(sys.argv[1:])
