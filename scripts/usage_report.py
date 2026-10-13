"""Usage report for opentextdetector.com from the Worker's Analytics Engine dataset `otd_usage` (app/atlas-cloud/usage.ts).

Needs a Cloudflare API token with the "Account Analytics: Read" permission, saved (not printed) in
~/.config/open-text-detector/analytics-token. Create it in the dashboard: My Profile -> API Tokens -> Create Token ->
Custom token -> Permissions: Account / Account Analytics / Read. Bots are left out unless --bots is given.
Counts are weighted by Analytics Engine's sampling (_sample_interval), so they stay correct under heavy traffic.
Usage: uv run -q --python 3.12 scripts/usage_report.py [--days 7] [--bots] [--top 15]
"""
import json, sys, urllib.request
from pathlib import Path

ACCOUNT = '6147916993a1cbfa4e3898aaeed27ae3'
TOKEN_FILE = Path.home() / '.config/open-text-detector/analytics-token'


def sql(query):
    token = TOKEN_FILE.read_text().strip()
    req = urllib.request.Request(f'https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/analytics_engine/sql',
                                 data=query.encode(), headers={'Authorization': 'Bearer ' + token}, method='POST')
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())['data']


def main(argv):
    days = int(argv[argv.index('--days') + 1]) if '--days' in argv else 7
    top = int(argv[argv.index('--top') + 1]) if '--top' in argv else 15
    where = f"timestamp > NOW() - INTERVAL '{days}' DAY" + ('' if '--bots' in argv else " AND blob6 != 'bot'")
    if not TOKEN_FILE.exists(): sys.exit(f'No token: create one with Account Analytics: Read and save it to {TOKEN_FILE}')
    n = "SUM(_sample_interval)"
    print(f'opentextdetector.com, last {days} days' + (' (bots included)' if '--bots' in argv else ' (bots excluded)'))
    print('\nPer day: page views, papers opened, visitors')
    for r in sql(f"SELECT toStartOfDay(timestamp) AS day, SUM(IF(blob1='page', _sample_interval, 0)) AS pages, SUM(IF(blob1='paper', _sample_interval, 0)) AS papers, COUNT(DISTINCT blob10) AS visitors FROM otd_usage WHERE {where} GROUP BY day ORDER BY day"):
        print(f"  {r['day'][:10]}  {int(r['pages']):>7,} pages  {int(r['papers']):>7,} papers  {int(r['visitors']):>6,} visitors")
    sections = [('Pages', "blob2", "blob1='page'"), ('Countries', 'blob3', '1=1'), ('Institutions and networks', 'blob5', '1=1'),
                ('Cities', 'blob4', '1=1'), ('Referring sites', 'blob8', "blob1='page' AND blob8 != ''"), ('Devices', 'blob6', '1=1'),
                ('Browsers', 'blob7', '1=1'), ('Languages', 'blob9', '1=1'), ('Papers opened', 'blob2', "blob1='paper'"),
                ('List use (sort, mode, filters)', 'blob11', "blob1='browse'"), ('Events', 'blob1', '1=1')]
    for title, col, cond in sections:
        rows = sql(f"SELECT {col} AS k, {n} AS c, COUNT(DISTINCT blob10) AS v FROM otd_usage WHERE {where} AND {cond} GROUP BY k ORDER BY c DESC LIMIT {top}")
        print(f'\n{title}  (requests · visitors)')
        for r in rows: print(f"  {int(r['c']):>7,} · {int(r['v']):>5,}  {r['k'] or '(none)'}")


if __name__ == '__main__':
    main(sys.argv[1:])
