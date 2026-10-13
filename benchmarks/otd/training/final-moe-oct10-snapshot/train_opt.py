"""Overnight sweep trainer: train_short.py recipe with command-line knobs. Writes ROOT/sweeps/<tag>/."""
from runtime import ROOT, require_space
from pathlib import Path
import os
import argparse, json, gzip, hashlib, random, shutil, time, math, traceback
from collections import Counter
import numpy as np
import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from safetensors.torch import save_file, load_file
from data import encode_example
import modeling_sweep
from modeling_sweep import Detector, collate, loss, training_loss
from adapters_short import attach_lora, parameter_groups


SHORT_SPAN_OVERSAMPLE = False  # set by --short-span-oversample
FSDP = False  # set by --fsdp: FSDP2 sharding over the data-parallel ranks (MoE on 2 A100s)


class ShardOptimizer:
    """The usual optimizer, run on each rank's local shard of the FSDP2 parameters.

    AdamW (8-bit or not) is elementwise, so a rank can update its own shard with no communication. bitsandbytes cannot
    step DTensors (and its kernels reject FSDP's padded shard views), so each shard gets its own contiguous BF16 copy;
    gradients are handed over before each step and the updated values copied back after. Only LoRA and head weights
    are trainable, so the copies are small. 8-bit quantization blocks are formed per shard instead of per whole tensor."""

    def __init__(self, groups, kind):
        self.pairs = []; pg = []
        for g in groups:
            ps = []
            for p in g['params']:
                local = p._local_tensor if hasattr(p, '_local_tensor') else p.data
                if local.numel() == 0:  # small tensors (e.g. 2-element head biases) leave some ranks an empty shard
                    continue
                q = torch.nn.Parameter(local.detach().clone().contiguous(), requires_grad=True)
                self.pairs.append((local, p, q)); ps.append(q)
            pg.append({**g, 'params': ps})
        self.opt = bnb.optim.AdamW8bit(pg, weight_decay=.01) if kind == 'adamw8bit' else torch.optim.AdamW(pg, weight_decay=.01)
        self.param_groups = self.opt.param_groups

    @torch.no_grad()
    def step(self):
        for local, p, q in self.pairs:
            g = p.grad
            q.grad = None if g is None else (g._local_tensor if hasattr(g, '_local_tensor') else g).contiguous().to(q.dtype)
        self.opt.step()
        for local, p, q in self.pairs:
            local.copy_(q)

    def zero_grad(self, set_to_none=True):
        for local, p, q in self.pairs:
            p.grad = None; q.grad = None

    @torch.no_grad()
    def refresh(self):
        """Copy the shards into the optimizer's copies (after the shards were overwritten, e.g. a resume)."""
        for local, p, q in self.pairs:
            q.copy_(local)

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, s):
        self.opt.load_state_dict(s)
SAVE_STAGE1 = not str(ROOT).startswith('/workspace/')  # H200 volume quota is tight; the single stage-1 checkpoint is never evaluated


def save(p, d):
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(p)


def rows(root, key, manifest, fraction, seed):
    b = gzip.decompress((root / 'prepared-v2' / f'{key}.jsonl.gz').read_bytes())
    assert hashlib.sha256(b).hexdigest() == manifest['files'][key]['sha256']
    rs = [json.loads(l) for l in b.splitlines()]
    if not key.startswith('stage'):
        return rs
    target = {k: round(v * fraction) for k, v in manifest['counts'][key].items()}; seen = Counter(); short = []
    for row in rs:
        if seen[row['dataset']] < target[row['dataset']]:
            short.append(row); seen[row['dataset']] += 1
    assert dict(seen) == target
    if SHORT_SPAN_OVERSAMPLE and key.startswith('stage2'):
        short = oversample_short_spans(short, random.Random(seed * 7919 + sum(map(ord, key))))
    random.Random(seed * 1000 + sum(map(ord, key))).shuffle(short)
    return short


def ai_span_chars(row):
    regions = row.get('regions') or []
    ai = [r['end'] - r['start'] for r in regions if r['label'] == 1]
    human = sum(r['end'] - r['start'] for r in regions if r['label'] == 0)
    return min(ai) if ai and human else None


def oversample_short_spans(rows, rng):
    """Swap paper windows whose AI span is >=600 chars for copies of existing windows with shorter spans.

    Same row count and dataset mix; no new text is created."""
    papers = [i for i, r in enumerate(rows) if r['dataset'] == 'papers' and r.get('supervision') != 'document_only']
    short = [rows[i] for i in papers if (ai_span_chars(rows[i]) or 10**9) < 600]
    long_ = [i for i in papers if (ai_span_chars(rows[i]) or 0) >= 600]
    if not short:
        return rows
    out = list(rows)
    for i in long_:
        out[i] = dict(rng.choice(short)); out[i]['oversampled_short_span'] = True
    return out


def batches(rs, tok, kind, stage, size):
    for i in range(0, len(rs), size):
        yield collate([encode_example(r, tok, kind, stage) for r in rs[i:i + size]], tok.pad_token_id)


def place_layers(model):
    """Split a model too big for one GPU across all visible GPUs by whole decoder layers (balanced), heads on GPU 0.
    Batches are built on cuda:0, so the heads and the loss stay there; accelerate hooks move activations between GPUs."""
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    n = torch.cuda.device_count(); cap = {i: f'{int(torch.cuda.get_device_properties(i).total_memory / 2**30 * .85)}GiB' for i in range(n)}
    layer_types = list({m.__class__.__name__ for name, m in model.backbone.named_modules() if name.startswith('layers.') and name.count('.') == 1})
    mem = get_balanced_memory(model, max_memory=cap, no_split_module_classes=layer_types, low_zero=False)
    dm = infer_auto_device_map(model, max_memory=mem, no_split_module_classes=layer_types)
    for h in ('token_head', 'segment_head', 'mixed_head', 'document_head'):
        dm[h] = 0
    assert set(dm.values()) <= set(range(n)) and len(set(dm.values())) == n, dm
    dispatch_model(model, device_map=dm)
    return {k: v for k, v in dm.items()}


def checkpoint(model, path, write=True):
    """Adapter weights as one unsharded file. Under FSDP every rank must call this (gathering is collective); only rank 0 writes."""
    tensors = {n: (p.full_tensor() if hasattr(p, 'full_tensor') else p).detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
               for n, p in model.named_parameters() if p.requires_grad}
    if write:
        tmp = path.with_suffix('.tmp'); save_file(tensors, str(tmp)); tmp.replace(path)


def restore(model, path):
    d = load_file(str(path)); assert set(d) == {n for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in d:
                p.copy_(d[n])


@torch.inference_mode()
def validate(model, tok, rs, stage, size):
    model.eval(); total = 0; n = 0
    for b in batches(rs, tok, model.kind, stage, size):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(b); v, _ = loss(o, b, stage)
        assert torch.isfinite(v); total += float(v) * b['logical_size']; n += b['logical_size']
    return total / n


class DivergenceGuard:
    """Flags a training-loss blow-up within ~20-30 steps of its onset.

    Per stage: exponential moving average of step loss (half-life 50 steps); after 150 steps it tracks the
    lowest EMA seen and trips once the EMA stays above 2x that minimum + 0.1 for 30 consecutive steps.
    Replayed on every H200 sweep log (Oct 4-5): it caught all four runs that blew up (moe-A-full, curve-4b-B,
    curve-9b-B, curve-9b-B2) and raised no alarm on the ~30 healthy ones.
    """
    HALF_LIFE, ARM, RATIO, ADD, PATIENCE = 50, 150, 2., .1, 30

    def __init__(self):
        self.stage = None

    def update(self, stage, loss):
        if stage != self.stage:
            self.stage, self.ema, self.min, self.n, self.over = stage, None, None, 0, 0
        a = 1 - .5 ** (1 / self.HALF_LIFE); self.n += 1
        self.ema = loss if self.ema is None else self.ema + a * (loss - self.ema)
        if self.n < self.ARM:
            return False
        self.min = self.ema if self.min is None else min(self.min, self.ema)
        self.over = self.over + 1 if self.ema > self.RATIO * self.min + self.ADD else 0
        return self.over >= self.PATIENCE

    def state(self):
        return {k: getattr(self, k, None) for k in ['stage', 'ema', 'min', 'n', 'over']}

    def load(self, s):
        for k, v in (s or {}).items():
            setattr(self, k, v)


class _NoWandb:
    summary = {}
    url = None
    id = None

    def log(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


def main(a):
    require_space(); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    import torch.distributed as dist
    world = int(os.environ.get('WORLD_SIZE', '1')); rank = int(os.environ.get('RANK', '0')); local = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        torch.cuda.set_device(local); dist.init_process_group('nccl')
    is0 = rank == 0
    put = save if is0 else (lambda *x, **k: None)
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
    spec = next(m for m in json.loads((ROOT / 'models.json').read_text()) if m['name'] == a.model)
    root = ROOT / 'runs' / a.model; out = ROOT / 'sweeps' / a.tag; out.mkdir(parents=True, exist_ok=True)
    assert a.resume or not (out / 'run.json').exists(), 'Existing sweep run: do not restart (use --resume)'
    global SHORT_SPAN_OVERSAMPLE
    SHORT_SPAN_OVERSAMPLE = a.short_span_oversample
    modeling_sweep.W['sentence'] = a.sentence_weight
    manifest = json.loads((root / 'prepared-v2/manifest.json').read_text())
    total_rows = sum(round(v * a.fraction) for k in ['stage1-epoch0'] + [f'stage2-epoch{e}' for e in range(a.stage2_epochs)] for v in manifest['counts'][k].values())
    size = a.micro_batch; A = 32 // size
    assert 32 % size == 0 and A % world == 0, 'effective batch 32 must split evenly into micro-batches across GPUs'
    cfg = {'kind': spec['kind'], 'training_fraction': a.fraction, 'total_training_examples': total_rows, 'seed': a.seed,
           'learning_rate': a.lr, 'head_learning_rate': a.head_lr, 'schedule': a.schedule, 'warmup_fraction': a.warmup,
           'micro_batch': size, 'physical_microbatch': size, 'effective_batch': 32, 'data_parallel_gpus': world,
           'stages': {'1': {'epochs': 1}, '2': {'epochs': a.stage2_epochs}}, 'loss_weights': dict(modeling_sweep.W),
           'lora': {'rank': spec.get('lora_rank', 128), 'expert_rank': spec.get('expert_rank', 128), 'alpha': 32},
           'checkpoint_every_steps': a.ckpt_every, 'short_span_oversample': a.short_span_oversample, 'source_tokens_max': 510, 'precision': 'BF16',
           'trainable_dtype': 'bfloat16', 'checkpoint_dtype': 'bfloat16', 'optimizer': a.optimizer}
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    asset = ROOT / 'assets' / a.model; tok = AutoTokenizer.from_pretrained(asset, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    put(out / 'status.json', {'state': 'loading_model'})
    model = Detector.load_base({'repo': str(asset), 'revision': None}, cfg['kind']); coverage = attach_lora(model, cfg)
    if a.fast_moe:  # unmerged expert LoRA over the routed tokens (same parameters and adapter files as peft)
        import fast_moe
        cfg['fast_moe_experts'] = fast_moe.patch(model); assert cfg['fast_moe_experts'] or not getattr(model.backbone.config, 'num_experts', 0)
    if a.pack:  # rows of a micro-batch run as one packed sequence (packing.py); exact same outputs, no padding compute
        import packing
        packing.enable(model); cfg['packed'] = {'bucket': packing.BUCKET, 'attention': packing.NAME}
    import importlib.util
    cfg['causal_conv1d'] = getattr(importlib.import_module('causal_conv1d'), '__version__', '?') if importlib.util.find_spec('causal_conv1d') else None
    assert all(p.dtype == torch.bfloat16 for p in model.parameters() if p.requires_grad)
    global FSDP
    FSDP = a.fsdp
    if FSDP:
        assert world > 1 and not a.model_parallel and a.on_diverge != 'rollback', '--fsdp needs torchrun with 2+ ranks; rollback is not supported'
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        from torch.distributed.device_mesh import init_device_mesh
        if not a.no_grad_ckpt:
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(model.backbone, 'enable_input_require_grads'):
            model.backbone.enable_input_require_grads()
        mesh = init_device_mesh('cuda', (world,)); mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        all_layers = list(model.backbone.layers); n_l = len(all_layers)
        if a.fsdp_no_ckpt_layers and not a.no_grad_ckpt:  # selective activation checkpointing: N evenly spaced layers keep activations
            nock = {round(j * n_l / a.fsdp_no_ckpt_layers) for j in range(a.fsdp_no_ckpt_layers)}
            for i_ in nock:
                assert hasattr(all_layers[i_], 'gradient_checkpointing'), 'decoder layers lack a per-layer gradient_checkpointing flag'
                all_layers[i_].gradient_checkpointing = False
        # --fsdp-keep N: N evenly spaced layers stay whole on every GPU (no all-gather), the rest are sharded.
        keep_idx = {round(j * n_l / a.fsdp_keep) for j in range(a.fsdp_keep)} if a.fsdp_keep else set()
        kept = [l_ for i_, l_ in enumerate(all_layers) if i_ in keep_idx]; layers = [l_ for i_, l_ in enumerate(all_layers) if i_ not in keep_idx]
        for l_ in kept:
            l_.cuda()
        kept_params = {p_ for l_ in kept for p_ in l_.parameters()}
        for i_, layer in enumerate(layers):
            # The last layer stays unsharded after forward: backward needs it again straight away.
            fully_shard(layer, mesh=mesh, mp_policy=mp, reshard_after_forward=a.fsdp_reshard and i_ < len(layers) - 1)
        fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=False, **({'ignored_params': kept_params} if kept_params else {}))
        # Sum (not average) gradients over ranks: each rank's loss is already weighted by its share of the 32-row batch.
        divide_set = [m_ for m_ in model.modules() if hasattr(m_, 'set_gradient_divide_factor')]
        for m_ in divide_set:
            m_.set_gradient_divide_factor(1.0)
        assert divide_set, 'this torch has no FSDPModule.set_gradient_divide_factor'
        replicated_trainable = [p_ for p_ in kept_params if p_.requires_grad]
        if a.fsdp_prefetch:
            for i_, layer in enumerate(layers):
                layer.set_modules_to_forward_prefetch(layers[i_ + 1:i_ + 1 + a.fsdp_prefetch])
                layer.set_modules_to_backward_prefetch(layers[max(0, i_ - a.fsdp_prefetch):i_][::-1])
        cfg['fsdp'] = {'world': world, 'reshard_after_forward': a.fsdp_reshard, 'prefetch': a.fsdp_prefetch, 'physical_microbatch': a.phys,
                       'replicated_layers': sorted(keep_idx), 'gradient_reduction': 'sum', 'layers_without_activation_checkpointing': a.fsdp_no_ckpt_layers, 'decoder_layers': n_l,
                       'optimizer_on_shards': True, 'grad_checkpointing': not a.no_grad_ckpt}
        model.physical_microbatch = a.phys  # one forward per micro-batch keeps the collective count equal on every rank
    else:
        if a.model_parallel and torch.cuda.device_count() > 1:
            assert world == 1, '--model-parallel splits one model across the visible GPUs; it cannot be combined with DDP'
            coverage['device_map'] = place_layers(model)
        else:
            model = model.cuda()
        model.physical_microbatch = size
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if a.fsdp_no_ckpt_layers:  # same selective checkpointing as the FSDP path (H200s have the memory for it)
            all_layers = list(model.backbone.layers); n_l = len(all_layers)
            for i_ in {round(j * n_l / a.fsdp_no_ckpt_layers) for j in range(a.fsdp_no_ckpt_layers)}:
                assert hasattr(all_layers[i_], 'gradient_checkpointing'), 'decoder layers lack a per-layer gradient_checkpointing flag'
                all_layers[i_].gradient_checkpointing = False
            cfg['layers_without_activation_checkpointing'] = a.fsdp_no_ckpt_layers
        if hasattr(model.backbone, 'enable_input_require_grads'):
            model.backbone.enable_input_require_grads()
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    if not FSDP:
        replicated_trainable = []

    def fsdp_finish_grads(max_norm):
        """FSDP: sum the replicated layers' gradients over ranks, then clip by the global norm of all gradients."""
        if replicated_trainable:
            flat = torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_)).float().reshape(-1) for p_ in replicated_trainable])
            dist.all_reduce(flat); o = 0
            for p_ in replicated_trainable:
                n_ = p_.numel(); p_.grad = flat[o:o + n_].view_as(p_).to(p_.dtype); o += n_
        rep_ids = {id(p_) for p_ in replicated_trainable}
        sh = torch.zeros((), device='cuda', dtype=torch.float64); rp_ = torch.zeros((), device='cuda', dtype=torch.float64)
        for p_ in trainable:
            if p_.grad is None:
                continue
            g = p_.grad._local_tensor if hasattr(p_.grad, '_local_tensor') else p_.grad
            if id(p_) in rep_ids:
                rp_ += g.float().pow(2).sum()
            else:
                sh += g.float().pow(2).sum()
        dist.all_reduce(sh)
        norm = float((sh + rp_).sqrt()); coef = max_norm / (norm + 1e-6)
        if coef < 1:
            for p_ in trainable:
                if p_.grad is not None:
                    (p_.grad._local_tensor if hasattr(p_.grad, '_local_tensor') else p_.grad).mul_(coef)
        return norm

    def ckpt(path):
        if FSDP:
            checkpoint(model, path, write=is0)
        elif is0:
            checkpoint(model, path)

    def sync_params():
        if world > 1 and not FSDP:  # FSDP shards stay consistent by construction
            with torch.no_grad():
                for p_ in trainable:
                    dist.broadcast(p_.data, 0)

    def reduce_grads():
        """Sum micro-batch gradients across GPUs in FP32; parameters unused on a rank (idle experts) contribute zeros."""
        if world == 1 or FSDP:  # FSDP reduce-scatters gradients itself on the last micro-batch
            return
        flat = torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_)).float().reshape(-1) for p_ in trainable])
        dist.all_reduce(flat); o = 0
        for p_ in trainable:
            n_ = p_.numel(); p_.grad = flat[o:o + n_].view_as(p_).to(p_.dtype); o += n_

    def total(*vals):
        t = torch.tensor(vals, dtype=torch.float64, device='cuda')
        if world > 1:
            dist.all_reduce(t)
        return t.tolist()

    def max_all(v):
        t = torch.tensor([v], dtype=torch.float64, device='cuda')
        if world > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    sync_params()
    put(out / 'run.json', {'config': cfg, 'model': spec, 'assets': str(asset), 'tag': a.tag,
                           'data_manifest_sha256': hashlib.sha256((root / 'prepared-v2/manifest.json').read_bytes()).hexdigest(),
                           'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}})
    wr = _NoWandb()
    if is0:
        import wandb
        settings = wandb.Settings(console='off', disable_code=True, disable_git=True, save_code=False, disable_job_creation=True,
                                  x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True)
        prior = json.loads((out / 'wandb-tracking.json').read_text()) if a.resume and (out / 'wandb-tracking.json').exists() else None
        wr = wandb.init(entity='rigg-alice0', project='pangram-text-classifiers', group='text-classifiers',
                        name=f'{a.model}-{a.tag}-20261004', config={}, settings=settings, dir=str(out),
                        **({'id': prior['run_id'], 'resume': 'allow'} if prior else {}))
        put(out / 'wandb-tracking.json', {'run_id': wr.id, 'url': wr.url})
    selection = rows(root, 'selection-windows', manifest, a.fraction, a.seed)[::3]  # fixed third keeps validation cheap

    @torch.no_grad()
    def validate_shared(stage):
        model.eval(); tot = 0.; cnt = 0; mine = selection[rank::world]
        nb = math.ceil(len(mine) / size)
        if FSDP:  # every rank must run the same number of forwards; extra ones repeat a batch with weight 0
            nb = int(max_all(nb))
        for i_ in range(nb):
            part = mine[i_ * size:(i_ + 1) * size]; w_ = 1.
            if not part:
                part, w_ = mine[:size], 0.
            b = collate([encode_example(r, tok, model.kind, stage) for r in part], tok.pad_token_id)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o_ = model(b); v_, _ = loss(o_, b, stage)
            assert torch.isfinite(v_); tot += w_ * float(v_) * b['logical_size']; cnt += w_ * b['logical_size']
        tot, cnt = total(tot, cnt)
        return tot / cnt

    history = []; started = time.time(); global_step = 0; processed = 0; done_rows = 0; bench = {}; step_losses = []
    sd = Path(a.state_dir) / a.tag if a.state_dir else out  # resume state can live off the quota-limited disk
    sd.mkdir(parents=True, exist_ok=True)
    rp = sd / (f'resume-state-rank{rank}.pt' if FSDP else 'resume-state.pt'); state = None; last_save = time.time()  # FSDP: one file per rank (local shards)
    guard = DivergenceGuard(); snaps = sd / 'snapshots'  # rolling adapter snapshots for rollback after a blow-up
    skip = set(); lr_scale = 1.; rollbacks = 0; steplog = []; snap_steps = []
    full_snaps = a.on_diverge == 'rollback'  # rollback needs optimizer state and data position, not just adapters

    def apply_state(st):
        nonlocal history, global_step, processed, done_rows, started, skip, lr_scale, rollbacks
        with torch.no_grad():
            for n_, p_ in model.named_parameters():
                if n_ in st['trainable']:
                    dst = getattr(p_, '_local_tensor', p_)
                    dst.copy_(st['trainable'][n_].to(dst.device, dst.dtype))
        history = st['history']; global_step = st['global_step']; processed = st['processed']; done_rows = st['done_rows']
        started = time.time() - st['elapsed']; guard.load(st.get('guard')); guard.over = 0
        skip = {tuple(x) for x in st.get('skip', [])}; lr_scale = st.get('lr_scale', 1.); rollbacks = st.get('rollbacks', 0)
        random.setstate(st['rng']['python']); np.random.set_state(st['rng']['numpy']); torch.set_rng_state(st['rng']['torch'])

    if a.resume and rp.exists():
        state = torch.load(rp, map_location='cpu', weights_only=False); apply_state(state)
        if FSDP:
            assert max_all(global_step) == -max_all(-global_step), 'per-rank resume files are from different steps'
        if snaps.exists():
            snap_steps = sorted(int(p_.name[4:9]) for p_ in snaps.glob('step*.pt'))
        if is0:
            print(json.dumps({'event': 'resumed', 'stage': state['stage'], 'epoch': state['epoch'], 'batch': state['batch'], 'optimizer_steps': global_step}), flush=True)

    def save_resume(stage, epoch, batch, running, best, bestpath, opt, sched, path=None):
        if not is0 and not FSDP:
            return
        path = path or rp
        snap = {'trainable': {n_: getattr(p_, '_local_tensor', p_).detach().to('cpu', torch.bfloat16) for n_, p_ in model.named_parameters() if p_.requires_grad},
                'opt': opt.state_dict(), 'sched': sched.state_dict(), 'stage': stage, 'epoch': epoch, 'batch': batch, 'running': running,
                'best': best, 'bestpath': str(bestpath) if bestpath else None, 'history': history, 'global_step': global_step,
                'processed': processed, 'done_rows': done_rows, 'elapsed': time.time() - started, 'guard': guard.state(),
                'skip': sorted(skip), 'lr_scale': lr_scale, 'rollbacks': rollbacks,
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}}
        tmp = path.with_suffix('.tmp'); torch.save(snap, tmp); tmp.replace(path)

    class _Stop(Exception):
        pass

    class _Diverged(Exception):
        pass

    class _Rollback(Exception):
        pass
    try:
        while True:
            try:
                for stage in [1, 2]:
                    if state is not None and stage < state['stage']:
                        continue
                    epochs = cfg['stages'][str(stage)]['epochs']; best = float('inf'); bestpath = None
                    if FSDP:  # validation leaves the root module gathered (no backward follows); the optimizer must see the sharded parameters
                        for m_ in model.modules():
                            if hasattr(m_, 'reshard'):
                                m_.reshard()
                        assert all(hasattr(p_, '_local_tensor') for p_ in model.parameters() if p_.requires_grad and not any(p_ is r_ for r_ in replicated_trainable)), 'parameters not sharded before optimizer setup'
                    opt = (ShardOptimizer(parameter_groups(model, cfg), a.optimizer) if FSDP
                           else bnb.optim.AdamW8bit(parameter_groups(model, cfg), weight_decay=.01) if a.optimizer == 'adamw8bit'
                           else torch.optim.AdamW(parameter_groups(model, cfg), weight_decay=.01))
                    total_steps = sum(math.ceil(round(manifest['files'][f'stage{stage}-epoch{e}']['rows'] * a.fraction) / 32) for e in range(epochs))
                    warm = math.ceil(a.warmup * total_steps); inner = getattr(opt, 'opt', opt)  # the scheduler drives the real optimizer
                    sched = (get_cosine_schedule_with_warmup(inner, warm, total_steps) if a.schedule == 'cosine'
                             else get_constant_schedule_with_warmup(inner, warm))
                    base0 = list(sched.base_lrs)
                    if state is not None and stage == state['stage']:
                        opt.load_state_dict(state['opt']); sched.load_state_dict(state['sched'])
                        best = state['best']; bestpath = Path(state['bestpath']) if state['bestpath'] else None
                    sched.base_lrs = [b_ * lr_scale for b_ in base0]  # learning rate cut after repeated rollbacks
                    for g_, b_, f_ in zip(opt.param_groups, sched.base_lrs, sched.lr_lambdas):
                        g_['lr'] = b_ * f_(sched.last_epoch)
                    stage_t0 = time.time(); stage_rows0 = done_rows
                    for epoch in range(epochs):
                        if state is not None and stage == state['stage'] and epoch < state['epoch']:
                            continue
                        start_i, running = 0, 0
                        if state is not None:
                            start_i, running = state['batch'], state['running']; state = None
                        rs = rows(root, f'stage{stage}-epoch{epoch}', manifest, a.fraction, a.seed)
                        model.train(); opt.zero_grad(set_to_none=True); n = math.ceil(len(rs) / size); epochstart = time.time()
                        for g0 in range(start_i, n, A):
                            if (stage, epoch, g0) in skip:  # batches dropped by an auto-rollback
                                continue
                            members = list(range(g0, min(g0 + A, n))); gn = len(members); loss_sum = 0.; toks = 0
                            group = rs[g0 * size:(g0 + gn) * size]
                            if world > 1 and a.balance:
                                # Deal the group's rows to GPUs by estimated cost (snake order), equal row counts per GPU.
                                order = sorted(range(len(group)), key=lambda j: -len(group[j]['text'])); bins = [[] for _ in range(world)]
                                for pos, j in enumerate(order):
                                    r_ = pos % (2 * world); bins[r_ if r_ < world else 2 * world - 1 - r_].append(group[j])
                                mine = bins[rank]; micro = [mine[j:j + size] for j in range(0, len(mine), size)]
                            else:
                                micro = [rs[k * size:(k + 1) * size] for k in members if (k - g0) % world == rank]
                            micro = [mb for mb in micro if mb]; real = len(micro)
                            if FSDP:  # equal forward/backward count on every rank; padding micro-batches carry weight 0
                                need = int(max_all(real)); micro += [group[:1]] * (need - real)
                            for k_, mb in enumerate(micro):
                                if FSDP:
                                    model.set_requires_gradient_sync(k_ == len(micro) - 1)
                                b = collate([encode_example(r, tok, cfg['kind'], stage) for r in mb], tok.pad_token_id)
                                with torch.autocast('cuda', dtype=torch.bfloat16):
                                    pred = model(b); v, parts = training_loss(pred, b, stage)
                                if not torch.isfinite(v):
                                    raise RuntimeError('Nonfinite training loss')
                                w_ = len(mb) / len(group) if k_ < real else 0.  # equals 1/gn when micro-batches are full: same objective as single-GPU accumulation
                                (v * w_).backward(); loss_sum += float(v.detach()) * w_ * gn; toks += int(b['attention_mask'].sum()) if k_ < real else 0
                                del b, pred, v
                            reduce_grads()
                            loss_sum, toks = total(loss_sum, toks)
                            grad_norm = fsdp_finish_grads(1.) if FSDP else float(torch.nn.utils.clip_grad_norm_(trainable, 1.)); lr_now = max(g_['lr'] for g_ in opt.param_groups)  # rate this step actually uses
                            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); global_step += 1
                            step_losses.append(round(loss_sum / gn, 6)); running += loss_sum; processed += int(toks); done_rows += len(rs[g0 * size:(g0 + gn) * size])
                            tripped = guard.update(stage, loss_sum / gn)  # identical on every rank: the loss is all-reduced
                            steplog.append((global_step, stage, epoch, g0, guard.ema)); del steplog[:-4000]
                            if a.snapshot_every and global_step % a.snapshot_every == 0 and not tripped and (not full_snaps or g0 + A < n):
                                snaps.mkdir(exist_ok=True)
                                if full_snaps:
                                    save_resume(stage, epoch, g0 + A, running, best, bestpath, opt, sched, path=snaps / f'step{global_step:05d}.pt')
                                    snap_steps.append(global_step)
                                    for old in snap_steps[:-a.snapshot_keep]:
                                        if is0:
                                            (snaps / f'step{old:05d}.pt').unlink(missing_ok=True)
                                    del snap_steps[:-a.snapshot_keep]
                                else:
                                    ckpt(snaps / f'step{global_step:05d}-adapters.safetensors')
                                    if is0:
                                        for old in sorted(snaps.glob('step*-adapters.safetensors'))[:-a.snapshot_keep]:
                                            old.unlink()
                            if tripped and a.on_diverge == 'rollback' and rollbacks < a.max_rollbacks:
                                # Onset = first step after the loss EMA last sat near its minimum; drop a small window of
                                # batches around it and replay from the newest snapshot taken before that window.
                                thr = 1.25 * guard.min + .02; cur = [e for e in steplog if e[1] == stage]; onset = cur[0][0]
                                for e in reversed(cur):
                                    if e[4] is not None and e[4] <= thr:
                                        onset = e[0] + 1; break
                                onset = min(onset, global_step); lo_, hi_ = onset - a.skip_before, onset + a.skip_after
                                usable = [s_ for s_ in snap_steps if s_ < lo_]
                                if usable:
                                    target = usable[-1]; new_skip = {(e[1], e[2], e[3]) for e in steplog if lo_ <= e[0] < hi_}
                                    new_scale = lr_scale * (a.rollback_lr_factor if rollbacks >= 1 else 1.)
                                    info = {'event': 'rollback', 'rollback': rollbacks + 1, 'tripped_at_step': global_step, 'onset_step': onset,
                                            'skipped_steps': [lo_, hi_], 'skipped_batches': len(new_skip), 'restored_snapshot_step': target,
                                            'lr_scale': new_scale, 'loss_ema': round(guard.ema, 4), 'loss_ema_min': round(guard.min, 4)}
                                    if is0:
                                        print(json.dumps(info), flush=True)
                                        log_ = json.loads((out / 'rollbacks.json').read_text()) if (out / 'rollbacks.json').exists() else []
                                        save(out / 'rollbacks.json', log_ + [info]); wr.summary['rollbacks'] = rollbacks + 1
                                        if hasattr(wr, 'alert'):
                                            wr.alert(title=f'{a.tag} rolled back', text=json.dumps(info)[:900])
                                        for s_ in snap_steps:
                                            if s_ > target:
                                                (snaps / f'step{s_:05d}.pt').unlink(missing_ok=True)
                                    if world > 1:
                                        dist.barrier()
                                    st = torch.load(snaps / f'step{target:05d}.pt', map_location='cpu', weights_only=False)
                                    keep_skip = skip | new_skip; n_rb = rollbacks + 1
                                    apply_state(st); skip = keep_skip; lr_scale = new_scale; rollbacks = n_rb
                                    steplog[:] = [e for e in steplog if e[0] <= target]; snap_steps[:] = [s_ for s_ in snap_steps if s_ <= target]
                                    state = st; sync_params()
                                    raise _Rollback()
                            if tripped:
                                info = {'event': 'diverged', 'stage': stage, 'epoch': epoch, 'batch': g0 + gn, 'optimizer_steps': global_step,
                                        'loss_ema': round(guard.ema, 4), 'loss_ema_min': round(guard.min, 4), 'learning_rate': lr_now,
                                        'rollbacks': rollbacks, 'snapshots': sorted(p.name for p in snaps.glob('step*')), 'snapshot_dir': str(snaps)}
                                if is0:
                                    print(json.dumps(info), flush=True); save(out / 'divergence.json', info)
                                    wr.summary['diverged_at_step'] = global_step
                                    if hasattr(wr, 'alert'):
                                        wr.alert(title=f'{a.tag} diverged', text=json.dumps(info)[:900])
                                if a.on_diverge != 'warn':  # stop, or rollback with no usable snapshot / rollbacks used up
                                    raise _Diverged()
                                guard.over = -10 ** 9  # warn mode: report once per stage, keep training
                            due = bool(a.resume_every and time.time() - last_save > a.resume_every * 60 and g0 + A < n)
                            if FSDP:  # per-rank files must come from the same step: decide together, not by each rank's clock
                                due = bool(max_all(float(due)))
                            if due:
                                save_resume(stage, epoch, g0 + A, running, best, bestpath, opt, sched); last_save = time.time()
                            if a.ckpt_every and stage == 2 and global_step % a.ckpt_every == 0:
                                ckpt(out / f'step{global_step:05d}-adapters.safetensors')
                                put(out / 'step-checkpoints.json', sorted(p.name for p in out.glob('step*-adapters.safetensors')))
                            if is0:
                                status = {'state': 'training', 'stage': stage, 'epoch': epoch, 'batch': g0 + gn, 'batches': n, 'loss': loss_sum / gn,
                                          'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step, 'logical_rows': done_rows,
                                          'total_rows': total_rows, 'processed_tokens': processed, 'gpus': world,
                                          'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1),
                                          'grad_norm': round(grad_norm, 4), 'learning_rate': lr_now,
                                          'loss_ema': guard.ema and round(guard.ema, 4), 'loss_ema_min': guard.min and round(guard.min, 4),
                                          'rollbacks': rollbacks, 'lr_scale': lr_scale, 'skipped_batches': len(skip)}
                                save(out / 'status.json', status)
                                if global_step % 8 == 0:
                                    print(json.dumps(status), flush=True)
                                wr.log({'train/loss': status['loss'], 'progress/stage': stage, 'progress/epoch': epoch + 1,
                                        'progress/percent': 100 * done_rows / total_rows, 'progress/optimizer_step': global_step,
                                        'train/elapsed_seconds': status['elapsed_seconds'], 'train/grad_norm': grad_norm,
                                        'train/learning_rate': lr_now, 'train/loss_ema': guard.ema, 'train/rollbacks': rollbacks,
                                        'train/tokens_per_second': processed / max(1, status['elapsed_seconds'])})
                            if a.max_steps and global_step >= a.max_steps:
                                bench[f'stage{stage}'] = {'rows': done_rows - stage_rows0, 'seconds': time.time() - stage_t0}
                                raise _Stop()
                        sync_params()
                        score = validate_shared(stage) if not a.no_validate else float('nan')
                        rec = {'stage': stage, 'epoch': epoch, 'train_loss': running / n, 'selection_loss': score,
                               'epoch_seconds': time.time() - epochstart, 'optimizer_steps': global_step}
                        history.append(rec); put(out / 'history.json', history)
                        wr.log({'epoch/index': len(history), 'epoch/train_loss': rec['train_loss'], 'epoch/validation_loss': score, 'epoch/seconds': rec['epoch_seconds']})
                        cp = out / f'stage{stage}-epoch{epoch}-adapters.safetensors'
                        if stage == 2 or SAVE_STAGE1:
                            ckpt(cp)
                        if score < best or best == float('inf'):
                            best = score; bestpath = cp; put(out / f'stage{stage}-selection.json', {**rec, 'checkpoint': cp.name})
                    bench[f'stage{stage}'] = {'rows': done_rows - stage_rows0, 'seconds': time.time() - stage_t0}
                    if stage == 1 and epochs > 1:
                        restore(model, bestpath)
                    del opt, sched; torch.cuda.empty_cache()
                break
            except _Rollback:
                opt = sched = None; torch.cuda.empty_cache()
        if is0 or FSDP:
            rp.unlink(missing_ok=True)
        if is0:
            shutil.rmtree(snaps, ignore_errors=True)
        put(out / 'status.json', {'state': 'trained', 'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step, 'processed_tokens': processed,
                                  'gpus': world, 'stage_throughput': bench, 'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)})
        wr.summary['training_state'] = 'finished'; wr.log({'progress/percent': 100}); wr.finish()
    except _Diverged:
        # Clean exit (status 0) so wrappers do not auto-resume into the same blow-up; snapshots and resume state are kept.
        put(out / 'status.json', {'state': 'diverged', 'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step,
                                  'detail': 'divergence.json', 'gpus': world})
        wr.summary['training_state'] = 'diverged'; wr.finish(exit_code=0)
    except _Stop:
        put(out / 'status.json', {'state': 'benchmark_done', 'elapsed_seconds': time.time() - started, 'optimizer_steps': global_step, 'step_losses': step_losses,
                                  'gpus': world, 'stage_throughput': bench, 'max_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)})
        wr.finish()
    except Exception as e:
        put(out / 'status.json', {'state': 'failed', 'error': repr(e), 'elapsed_seconds': time.time() - started})
        wr.summary['training_state'] = 'failed'; wr.finish(exit_code=1); raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model'); p.add_argument('tag')
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--head-lr', type=float, default=2e-5)
    p.add_argument('--schedule', choices=['cosine', 'constant'], default='cosine'); p.add_argument('--warmup', type=float, default=.06)
    p.add_argument('--sentence-weight', type=float, default=.2); p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--fraction', type=float, default=.2); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt-every', type=int, default=0)
    p.add_argument('--optimizer', choices=['adamw8bit', 'adamw'], default='adamw8bit')
    p.add_argument('--short-span-oversample', action='store_true')
    p.add_argument('--resume', action='store_true'); p.add_argument('--resume-every', type=float, default=20.0)
    p.add_argument('--model-parallel', action='store_true', help='split one model across all visible GPUs by layers (MoE on 2 A100s); no DDP')
    p.add_argument('--fsdp', action='store_true', help='FSDP2: shard the model over the torchrun ranks, each rank trains on its share of every batch')
    p.add_argument('--fsdp-reshard', action=argparse.BooleanOptionalAction, default=True, help='free gathered layer weights after forward (saves memory)')
    p.add_argument('--fsdp-prefetch', type=int, default=0, help='explicitly prefetch the next N layers in forward and backward')
    p.add_argument('--phys', type=int, default=4096, help='FSDP: rows per physical forward (default: whole micro-batch in one forward)')
    p.add_argument('--no-grad-ckpt', action='store_true', help='FSDP: disable activation checkpointing (faster, more memory)')
    p.add_argument('--fsdp-no-ckpt-layers', type=int, default=0, help='FSDP: N evenly spaced layers skip activation checkpointing (faster, more memory)')
    p.add_argument('--fsdp-keep', type=int, default=0, help='FSDP: keep N evenly spaced decoder layers whole on every GPU (fewer all-gathers, more memory)')
    p.add_argument('--max-steps', type=int, default=0); p.add_argument('--no-validate', action='store_true')
    p.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pack', action='store_true', help='causal backbones: pack each physical micro-batch into one sequence (needs ccdshim on PYTHONPATH)')
    p.add_argument('--fast-moe', action='store_true', help='MoE: grouped-matmul experts with unmerged LoRA (fast_moe.py) instead of peft merging every expert weight')
    p.add_argument('--state-dir', default='')
    p.add_argument('--stage2-epochs', type=int, default=3, help='stage-2 passes (1 = single pass over epoch-0 data)')
    p.add_argument('--snapshot-every', type=int, default=100, help='rolling adapter snapshots (in the state dir) every N optimizer steps; 0 disables')
    p.add_argument('--snapshot-keep', type=int, default=3)
    p.add_argument('--on-diverge', choices=['stop', 'warn', 'rollback'], default='stop',
                   help='rollback: restore the newest snapshot before the blow-up, skip the batches around its onset and continue')
    p.add_argument('--max-rollbacks', type=int, default=3)
    p.add_argument('--skip-before', type=int, default=4, help='optimizer steps skipped before the detected onset')
    p.add_argument('--skip-after', type=int, default=12, help='optimizer steps skipped from the onset on')
    p.add_argument('--rollback-lr-factor', type=float, default=.5, help='learning-rate multiplier applied from the second rollback on')
    args = p.parse_args()
    try:
        main(args)
    except Exception as e:
        d = ROOT / 'sweeps' / args.tag; d.mkdir(parents=True, exist_ok=True)
        save(d / 'status.json', {'state': 'failed', 'error': repr(e)}); traceback.print_exc(); raise
