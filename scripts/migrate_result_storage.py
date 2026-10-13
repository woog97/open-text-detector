#!/usr/bin/env python3
"""Resumable, verified migration of legacy results (or local objects to S3).
Run from backend with ../scripts/migrate_result_storage.py; defaults to a dry run.
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from open_text_detector_backend.config import Settings
from open_text_detector_backend.db import Database
from open_text_detector_backend.result_store import ResultStore


def migrate(db, store, apply=False):
    totals = {'reports': 0, 'original_json_bytes': 0, 'compressed_bytes': 0}
    # Read one result at a time; never materialize the whole corpus.
    ids = db.all("SELECT id FROM scans WHERE status='completed' AND result IS NOT NULL ORDER BY id")
    for item in ids:
        row = db.one('SELECT id,result,result_storage FROM scans WHERE id=?', (item['id'],))
        ref = json.loads(row['result_storage']) if row['result_storage'] else None
        if ref and (ref['backend'] == store.settings.result_storage):
            if ref['backend'] == 'local' or ref.get('store_id') == store.store_id:
                continue
        original = store.get(row['result_storage']) if ref else json.loads(row['result'])
        totals['reports'] += 1
        totals['original_json_bytes'] += len(json.dumps(original).encode())
        if not apply:
            continue
        summary, reference = store.put(original)
        # Compare-and-swap; an interrupted run can be safely rerun.
        changed = db.execute('UPDATE scans SET result=?,result_storage=? WHERE id=? AND result=? AND result_storage IS ?',
                             (summary, reference, row['id'], row['result'], row['result_storage']))
        if changed != 1:
            raise RuntimeError('Report changed during migration; originals were not overwritten')
        totals['compressed_bytes'] += json.loads(reference)['bytes']
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    settings = Settings()
    source = settings.data_dir.resolve() / 'workspace.sqlite3'
    if not source.is_file():
        parser.error('Existing workspace database not found; check working directory and OTD_DATA_DIR')
    if args.apply:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = source.with_name(f'workspace-before-result-migration-{stamp}.sqlite3')
        backup.touch(mode=0o600, exist_ok=False)
        with sqlite3.connect(source) as src, sqlite3.connect(backup) as dst:
            src.backup(dst)
        print('Recovery database:', backup.name)
    db = Database(settings.data_dir)
    print(json.dumps(migrate(db, ResultStore(settings), args.apply)))
    print('Objects and recovery backups are retained. SQLite free pages are reusable; no automatic VACUUM.')


if __name__ == '__main__':
    main()
