"""Metrics-only sidecar for current-data trainer; does not import or alter the model."""
import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, '/tmp/pangram-wandb-vendor')
import wandb

ENTITY = 'rigg-alice0'
PROJECT = 'open-text-detector-classifiers'
GROUP = 'text-classifiers'


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    root = args.root
    receipt_path = root / 'wandb-tracking.json'
    receipt = read(receipt_path, {})
    identity = read(root / 'training-process.json')
    cfg = read(root / 'run/run.json')['config']
    manifest = read(root / 'prepared-v2/manifest.json')
    # Local-only schedule math. Neither configuration nor manifest is uploaded.
    schedule = []
    for stage in (1, 2):
        for epoch in range(cfg['stages'][str(stage)]['epochs']):
            rows = manifest['files'][f'stage{stage}-epoch{epoch}']['rows']
            micro = cfg['stages'][str(stage)].get('micro_batch', cfg['micro_batch'])
            schedule.append((stage, epoch, math.ceil(rows / micro)))
    total_batches = sum(x[2] for x in schedule)
    settings = wandb.Settings(
        console='off', disable_code=True, disable_git=True,
        save_code=False, disable_job_creation=True,
        x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
    )
    Path('/tmp/pangram-wandb').mkdir(exist_ok=True)
    run = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     id=receipt.get('run_id', uuid.uuid4().hex[:12]),
                     name=args.name, resume='allow', config={},
                     dir='/tmp/pangram-wandb', settings=settings)
    run.define_metric('progress/batch')
    run.define_metric('train/*', step_metric='progress/batch')
    run.define_metric('progress/percent', step_metric='progress/batch')
    run.define_metric('epoch/index')
    run.define_metric('epoch/*', step_metric='epoch/index')
    receipt.update(run_id=run.id, url=run.url, project=PROJECT, group=GROUP)
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'url': run.url}), flush=True)
    cursor_path = root / 'wandb-tracking-cursor.json'
    cursor = read(cursor_path, {'batch': -1, 'epochs': 0})
    offset = 0
    while True:
        with (root / 'training.log').open() as stream:
            stream.seek(offset)
            while True:
                pos = stream.tell()
                line = stream.readline()
                if not line or not line.endswith('\n'):
                    offset = pos
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get('state') != 'training':
                    continue
                before = 0
                for stage, epoch, count in schedule:
                    if (stage, epoch) == (item['stage'], item['epoch']):
                        break
                    before += count
                batch = before + item['batch'] + 1
                if batch <= cursor['batch']:
                    continue
                run.log({'progress/batch': batch,
                         'progress/percent': 100 * batch / total_batches,
                         'progress/stage': item['stage'],
                         'train/loss': item['loss'],
                         'train/elapsed_seconds': item['elapsed_seconds']})
                cursor['batch'] = batch
        history = read(root / 'run/history.json', [])
        for i in range(cursor['epochs'], len(history)):
            item = history[i]
            run.log({'epoch/index': i + 1,
                     'epoch/train_loss': item['train_loss'],
                     'epoch/validation_loss': item['selection_loss'],
                     'epoch/seconds': item['epoch_seconds']})
            cursor['epochs'] = i + 1
        cursor_path.write_text(json.dumps(cursor))
        state = read(root / 'run/status.json', {})
        run.summary['training_state'] = state.get('state', 'unknown')
        proc = Path('/proc', str(identity['pid']), 'stat')
        try:
            parts = proc.read_text().split()
            alive = parts[2] != 'Z' and parts[21] == str(identity['start_ticks'])
        except FileNotFoundError:
            alive = False
        if not alive:
            complete = state.get('state') == 'trained_calibration_pending'
            if complete:
                run.log({'progress/batch': total_batches, 'progress/percent': 100.0})
            run.summary['training_state'] = 'finished' if complete else 'failed'
            run.finish(exit_code=0 if complete else 1)
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
