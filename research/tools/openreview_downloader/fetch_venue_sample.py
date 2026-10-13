"""Download a seeded random sample of a venue's papers (accepted and rejected) in 50-PDF batches.

The selection is the full decision list shuffled with a fixed seed, so a larger --head
on a later run extends the same sample (existing PDFs are skipped, never re-requested).
Credentials come from the macOS Keychain entry used by queue.py; nothing is printed.
Usage: fetch_venue_sample.py --venue-id ICLR.cc/2024/Conference --out-dir DIR --head 50
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

from openreview_downloader import cli
from openreview_downloader.queue import credentials


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venue-id', required=True)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--head', required=True, type=int)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exclude-ids', type=Path, help='JSON list of forum ids to leave out (e.g. papers already in the archive)')
    a = p.parse_args()
    username, password = credentials(keychain=True)
    os.environ['OPENREVIEW_USERNAME'], os.environ['OPENREVIEW_PASSWORD'] = username, password
    original = cli.filter_selected

    def shuffled(selected, args):
        matched = sorted(original(selected, args), key=lambda item: item[0].id)
        random.Random(f'{a.venue_id}:{a.seed}').shuffle(matched)
        if a.exclude_ids:
            skip = set(json.loads(a.exclude_ids.read_text()))
            matched = [item for item in matched if item[0].id not in skip]   # relative order unchanged
        (a.out_dir / ('selection-excluding.json' if a.exclude_ids else 'selection.json')).write_text(json.dumps(
            {'venue_id': a.venue_id, 'seed': a.seed, 'order': [item[0].id for item in matched]}))
        return matched

    cli.filter_selected = shuffled
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cli.main(['all', '--venue-id', a.venue_id, '--out-dir', str(a.out_dir), '--head', str(a.head), '--batch-size', '50'])


if __name__ == '__main__':
    main()
