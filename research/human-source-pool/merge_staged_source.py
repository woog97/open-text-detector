"""Merge an independently curated source batch through the shared pool's checks."""
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import time
from collect_pool import atomic_json, digest, now
from expand_pool import add, connect


def validate_pair(pair, source):
    row, raw = pair['row'], pair['raw']
    assert row['source_id'] == raw['source_id'] == source['source_id']
    assert row['category'] == source['category']
    assert row['admission_status'] == 'quarantined_candidate' and row['training_eligible'] is False
    text = raw['record']['text']
    assert digest(text) == raw['raw_text_sha256'] == row['raw_text_sha256']
    assert 0 <= row['raw_start'] < row['raw_end'] <= len(text)
    assert text[row['raw_start']:row['raw_end']] == row['text']
    assert digest(row['text']) == row['passage_sha256']
    assert row['original_split'] == raw['record']['metadata']['official_split'] == 'train'
    assert row['source_revision'] == raw['source_revision'] == raw['record']['metadata']['archive_sha256']
    assert row['word_count'] == len(row['text'].split())
    bins = [(50,149),(150,399),(400,999),(1000,1500)]
    lo, hi = bins[row['length_bin']]
    assert lo <= row['word_count'] <= hi
    return row, raw


def merge(base, stage):
    started = time.monotonic()
    manifest = json.loads((stage/'package-manifest.json').read_text())
    payload = stage/'accepted-pairs.jsonl.gz'
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == manifest['package_sha256']
    plan_file = base/'pipeline/sampling-plan.json'
    assert hashlib.sha256(plan_file.read_bytes()).hexdigest() == manifest['plan_sha256']
    plan = json.loads(plan_file.read_text())
    source = next(s for s in plan['source_quotas'] if s['source_id'] == manifest['source_id'])
    pairs = [validate_pair(json.loads(line), source) for line in gzip.open(payload, 'rt')]
    assert len(pairs) == manifest['candidate_count']
    assert len({r['record_id'] for r, _ in pairs}) == len(pairs)
    assert digest('\n'.join(sorted(r['record_id'] for r, _ in pairs))) == manifest['record_ids_sha256']
    with (base/(source['source_id']+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect(base/'collection.sqlite3')
        try:
            db.execute('BEGIN IMMEDIATE')
            before = db.execute('SELECT count(*) FROM passages WHERE source=?',(source['source_id'],)).fetchone()[0]
            inserted = sum(add(db, row, raw, source['planned_passages']) for row, raw in pairs)
            after = db.execute('SELECT count(*) FROM passages WHERE source=?',(source['source_id'],)).fetchone()[0]
            assert after <= source['planned_passages'] and after == before + inserted
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
    receipt = {'source_id':source['source_id'], 'inserted':inserted, 'already_present_or_global_duplicate':len(pairs)-inserted,
               'source_count':after,'quota':source['planned_passages'],'package_sha256':manifest['package_sha256'],
               'all_offsets_hashes_tags_verified':True,'training_admitted':0,'elapsed_seconds':time.monotonic()-started,'updated_at':now()}
    atomic_json(stage/'merge-receipt.json',receipt)
    print(json.dumps(receipt))
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--stage',type=Path,required=True)
    a=p.parse_args();merge(a.base,a.stage)
