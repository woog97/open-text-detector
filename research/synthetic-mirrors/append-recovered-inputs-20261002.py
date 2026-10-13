"""Space-only append migration. No credentials, HTTP client, or paid requests."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone

BASE = Path('/tmp/pangram-luna-active-20261002')
RUN = BASE / 'run'
HUMAN_BASE = Path('/tmp/pangram-human-active-20261002')
MIGRATION = 'append-balanced-10000-sources-20261002T1420-v2'
SOURCES = ['govreport', 'imdb', 'writingprompts', 'opinrank', 'oanc_slate', 'oanc_icic', 'acl', 'ubuntu_irc', 'persuade']
CATEGORIES = {'govreport':'professional','imdb':'reviews','writingprompts':'creative','opinrank':'reviews',
              'oanc_slate':'news','oanc_icic':'professional','acl':'scientific','ubuntu_irc':'social','persuade':'essays'}

import sys
sys.path.insert(0,str(BASE/'pipeline'))
os.environ.update(LUNA_CHECKPOINT_ROOT=str(BASE), LUNA_BUCKET_ID='open-text-detector/training-storage', LUNA_BUCKET_PREFIX='workspace/synthetic-mirrors-luna-dollar-v1-recovered-20261002')
from durable_store import configured_store
STORE=configured_store()
assert STORE is not None

def sha(data): return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()
def canonical(data): return json.dumps(data, sort_keys=True, ensure_ascii=False)
def now(): return datetime.now(timezone.utc).isoformat()
def read_files(directory):
    paths = sorted(directory.glob('*.json'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(lambda p:(p.name,p.read_bytes()),paths))
def snapshot(directory): return {name:sha(data) for name,data in read_files(directory).items()}
def atomic(path, data):
    temp = path.with_name(path.name + '.' + MIGRATION + '.tmp')
    with temp.open('wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    temp.replace(path)
    STORE.persist(path)
def stopped():
    assert not (RUN/'stop-requested.json').exists(), 'Explicit stop flag requires reconciliation'
    p = json.loads((BASE / 'process.json').read_text()); path = Path('/proc') / str(p['pid']) / 'stat'
    if path.exists():
        fields = path.read_text().split()
        if fields[21] == str(p['start_ticks']) and fields[2] != 'Z': raise RuntimeError('Generator is still alive')
    if json.loads((RUN / 'status.json').read_text())['state'] != 'waiting_for_balanced_sources':
        raise RuntimeError('Expected naturally stopped balanced-source wait')

def append(db_path):
    if not db_path.is_file() or str(db_path).startswith('/data/'): raise RuntimeError('Use the validated local-/tmp Space backup')
    archive = RUN / 'migrations' / MIGRATION
    if archive.exists(): raise RuntimeError('Migration directory already exists; inspect prior outcome, never replay blindly')
    with (BASE / 'append-migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped()
        limits_bytes = (RUN / 'limits.json').read_bytes(); limits = json.loads(limits_bytes)
        assert limits['max_documents'] == 10000 and Decimal(str(limits['budget_usd'])) == Decimal('20')
        assert sum(limits['source_quotas'].values()) == 10000
        old_bytes = (BASE / 'input.jsonl').read_bytes()
        if not old_bytes.endswith(b'\n'): raise RuntimeError('Frozen input must have final JSONL delimiter')
        old_rows = [json.loads(x) for x in old_bytes.decode().split('\n') if x]
        index = {r['record_id']:r for r in old_rows}
        assert len(index) == len(old_rows)
        summary_bytes = (BASE / 'input-summary.json').read_bytes(); summary = json.loads(summary_bytes)
        manifest_bytes = (RUN / 'manifest.json').read_bytes(); manifest = json.loads(manifest_bytes)
        assert summary['sha256'] == sha(old_bytes) and summary['rows'] == len(old_rows)
        assert manifest['input_sha256'] == sha(canonical(old_rows))
        assert manifest['runner_sha256'] == sha((BASE/'pipeline/run_openrouter.py').read_bytes())
        assert manifest['core_sha256'] == sha((BASE/'pipeline/mirror_core.py').read_bytes())
        assert manifest['generator'] == json.loads((BASE/'pipeline/generator.json').read_text())
        ledger = {}; claimed = set(); confirmed = Decimal('0'); uncertain = Decimal('0'); states = Counter()
        for filename,data in read_files(RUN / 'calls').items():
            call = json.loads(data); ledger[filename] = sha(data)
            state = call.get('state'); states[state] += 1
            if state == 'complete': confirmed += Decimal(str(call['cost_usd']))
            elif state == 'interrupted_reserved_no_replay': uncertain += Decimal(str(call['reserved_usd']))
            else: raise RuntimeError('Unreconciled call state: ' + filename + ':' + str(state))
            if call['source_record_id'] not in index: raise RuntimeError('Paid parent missing from frozen queue')
            claimed.add(call['source_record_id'])
        assert len(ledger) >= 577 and states['interrupted_reserved_no_replay'] == 4
        assert confirmed.is_finite() and uncertain.is_finite() and Decimal('0') <= confirmed + uncertain < Decimal('20')
        assert uncertain == Decimal('0.003022625')
        assert confirmed >= Decimal('0.0865198625') - Decimal('0.000000000001')
        used = Counter(index[i]['source_id'] for i in claimed)
        status = json.loads((RUN / 'status.json').read_text())
        assert dict(used) == status['source_attempt_counts'] and len(claimed) == status['completed'] and 289 <= len(claimed) <= 10000
        assert all(n <= limits['source_quotas'].get(sid,0) for sid,n in used.items())
        record_hashes = snapshot(RUN / 'records')
        queued_unused = Counter(r['source_id'] for r in old_rows if r['record_id'] not in claimed)
        ids = set(index); parents = {r['parent_document_id'] for r in old_rows}; families = {r.get('provisional_family_id') or r['parent_document_id'] for r in old_rows}
        additions = []; added = Counter(); filtered = Counter(); existing_unused_reservations = {}; deficits = {}
        registry=json.loads((HUMAN_BASE / 'pipeline/source-registry.json').read_text())
        categories={r['id']:r['category'] for r in registry['sources']}
        sources=[sid for sid,n in limits['source_quotas'].items() if n>0 and sid!='asap2']
        db = sqlite3.connect('file:' + str(db_path) + '?mode=ro',uri=True,timeout=30)
        try:
            integrity = db.execute('PRAGMA quick_check').fetchall()
            if integrity != [('ok',)]: raise RuntimeError('Backup quick_check failed')
            for sid in sources:
                available_slots = max(0, limits['source_quotas'].get(sid,0) - used.get(sid,0))
                reserved = min(available_slots,queued_unused.get(sid,0)); existing_unused_reservations[sid] = reserved
                need = available_slots - reserved
                if not need: continue
                for (record_id,) in db.execute('SELECT id FROM passages WHERE source=? ORDER BY id',(sid,)):
                    if added[sid] >= need: break
                    if record_id in ids: filtered['existing_record'] += 1; continue
                    row = json.loads(db.execute('SELECT row FROM passages WHERE id=?',(record_id,)).fetchone()[0])
                    family = row.get('provisional_family_id') or row['parent_document_id']; parent = row['parent_document_id']
                    if parent in parents or family in families: filtered['existing_parent_or_family'] += 1; continue
                    text = row.get('text') or ''; wc = len(text.split())
                    if not 150 <= wc <= 1500: filtered['length'] += 1; continue
                    if sum(c.isalpha() for c in text) / max(1,len(text)) < .65: filtered['prose_fraction'] += 1; continue
                    assert row['source_id'] == sid and row['category'] == categories[sid]
                    assert sha(text) == row['passage_sha256'], ('passage_hash',sid)
                    assert row['offset_unit'] in ('unicode_codepoints','unicode_codepoints_in_preserved_extracted_text'), ('offset_unit',sid,row['offset_unit'])
                    assert row['word_count'] == wc, (sid, row['word_count'], wc)
                    raw_result = db.execute('SELECT raw FROM documents WHERE hash=?',(row['raw_text_sha256'],)).fetchone()
                    if raw_result is None: raise RuntimeError('Original missing for ' + record_id)
                    raw = json.loads(gzip.decompress(raw_result[0])); original = raw['record']['text']
                    assert sha(original) == row['raw_text_sha256']
                    assert 0 <= row['raw_start'] < row['raw_end'] <= len(original)
                    assert original[row['raw_start']:row['raw_end']] == text
                    assert row.get('training_eligible') is False and row.get('admission_status') == 'quarantined_candidate'
                    additions.append(row); added[sid] += 1; ids.add(record_id); parents.add(parent); families.add(family)
                deficits[sid] = need - added[sid]
        finally: db.close()
        if not additions:
            print(json.dumps({'state':'no_new_eligible_inputs','deficits':deficits})); return
        assert len(claimed) + sum(existing_unused_reservations.values()) + len(additions) <= 10000
        for sid,n in added.items(): assert used.get(sid,0) + existing_unused_reservations[sid] + n <= limits['source_quotas'][sid]
        # Every potentially failing source read happens before any frozen input is changed.
        stopped()
        assert (BASE / 'input.jsonl').read_bytes() == old_bytes
        assert (RUN / 'manifest.json').read_bytes() == manifest_bytes
        assert (RUN / 'limits.json').read_bytes() == limits_bytes
        assert snapshot(RUN / 'calls') == ledger and snapshot(RUN / 'records') == record_hashes
        archive.mkdir()
        shutil.copy2(BASE / 'input.jsonl',archive / 'previous-input.jsonl')
        shutil.copy2(BASE / 'input-summary.json',archive / 'previous-input-summary.json')
        shutil.copy2(RUN / 'manifest.json',archive / 'previous-manifest.json')
        shutil.copy2(RUN / 'limits.json',archive / 'limits.json')
        (archive / 'paid-ledger-hashes.json').write_text(json.dumps(ledger,indent=2))
        (archive / 'record-hashes.json').write_text(json.dumps(record_hashes,indent=2))
        rows = old_rows + additions
        new_bytes = old_bytes + ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in additions).encode()
        summary.update(rows=len(rows),sha256=sha(new_bytes),bytes=len(new_bytes))
        summary['source_counts'] = dict(Counter(r['source_id'] for r in rows))
        manifest['input_sha256'] = sha(canonical(rows))
        # Archive is sufficient to recover from any interrupted multi-file update.
        receipt = {'migration':MIGRATION,'state':'prepared','created_at':now(),'source_database':str(db_path),
           'original_input_sha256':sha(old_bytes),'input_sha256':sha(new_bytes),'old_rows':len(old_rows),'new_rows':len(rows),
           'added_sources':dict(added),'deficits':deficits,'existing_unused_slots_reserved':existing_unused_reservations,
           'filtered':dict(filtered),'confirmed_cost_usd':str(confirmed),'uncertain_reserve_usd':str(uncertain),
           'paid_call_count':len(ledger),'prior_attempts':len(claimed),'all_original_hashes_offsets_verified':True,
           'all_existing_parent_and_family_ids_excluded':True,'asap_inputs_added':0,'budget_usd':20,'max_documents':10000,
           'no_paid_requests_made':True,'original_input_prefix_preserved':True}
        (archive / 'receipt.json').write_text(json.dumps(receipt,indent=2))
        for archived in sorted(archive.iterdir()):
            if archived.is_file(): STORE.persist(archived)
        atomic(BASE / 'input.jsonl',new_bytes)
        atomic(BASE / 'input-summary.json',(json.dumps(summary,indent=2)+'\n').encode())
        atomic(RUN / 'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
        assert (BASE / 'input.jsonl').read_bytes().startswith(old_bytes)
        assert snapshot(RUN / 'calls') == ledger and snapshot(RUN / 'records') == record_hashes
        assert (RUN / 'limits.json').read_bytes() == limits_bytes
        stopped(); receipt.update(state='complete',completed_at=now(),paid_ledger_unchanged=True,limits_unchanged=True)
        atomic(archive / 'receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())
        print(json.dumps(receipt),flush=True)

# The parent-provided integrity-verified backup is inserted only after validation.
# append(Path('/tmp/VALIDATED_BACKUP_REQUIRED.sqlite3'))

db=Path('/tmp/pangram-human-active-20261002/checkpoints/cached-span-recovery-20261002T081635Z.sqlite3')
h=hashlib.sha256()
with db.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()=='2a5a2154f6790f5c1843a3cbea26f2d299b7a6ceb1af20a372d7c90b9be6aee9'
append(db)
