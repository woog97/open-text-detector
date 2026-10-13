# Snapshot: code of run `abl-noseg-nomix` (Oct 10, 2026)

This folder holds the byte-exact Python source of training run `abl-noseg-nomix`, the model that produces the scores on
[opentextdetector.com](https://opentextdetector.com) and is published on Hugging Face as
[`woog/open-text-detector-qwen36-35b-a3b-oct10`](https://huggingface.co/woog/open-text-detector-qwen36-35b-a3b-oct10)
(LoRA adapter plus heads on `Qwen/Qwen3.6-35B-A3B`, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`).

The trainer records the SHA-256 of every `*.py` file in its working directory in the run's `run.json`
(`code_sha256`). Each `.py` file here was recovered from a backup of the training machine or from this repository's git
history and checked against those hashes. Do not edit these files; change the maintained versions listed below instead.

The run was trained on one H200 GPU on 2026-10-10, 13:30–15:24 PDT (one pass over 72,000 stage-2 windows, 2,250
optimizer steps, no stage 1). The checkpoint that is served is `stage2-epoch0-adapters.safetensors` (the only
checkpoint; `stage2-selection.json` names it).

## Files recorded in `run.json` (`code_sha256`)

All 18 hashes match. "Maintained version" is where the file is developed in this repository; "same" means the
maintained file is still byte-identical to the snapshot.

| File | SHA-256 | Maintained version |
|---|---|---|
| `train_sweep.py` | `1dbb126e53895ddda6a2dee05135cf737f90157257762f2e26985dc69cf51a3e` | `benchmarks/otd/training/overnight-sweep-20261004/train_sweep.py` (differs) |
| `modeling_sweep.py` | `72a6cd36655692329be0ca5f0e83d95b6d1fad2259645b677ba2e4c28d15d461` | `benchmarks/otd/training/overnight-sweep-20261004/modeling_sweep.py` (differs) |
| `data.py` | `d8abef7c89a5606516c146d97c067f6cf9150430c1059fcb46105a14c2bae276` | `benchmarks/otd/training/backbone-launch-20261003/data.py` (differs) |
| `adapters_short.py` | `75ea7b01fd422c042cb707b4e534ade6f5476e060712c4804d60a50ba2e00472` | `benchmarks/otd/training/backbone-launch-20261003/adapters_short.py` (same) |
| `modeling.py` | `177c6bef4d0cf4d316d2a511a734f2473fd22c88b204807a0d0744e3001ffcd0` | `benchmarks/otd/training/backbone-launch-20261003/modeling.py` (same) |
| `fast_moe.py` | `7cdeaa464d6a9ab6ee048df4ee7e93b713021adb7b891a4df47465b93a5b0a61` | `benchmarks/otd/training/overnight-sweep-20261004/fast_moe.py` (same) |
| `gmm_triton.py` | `499d00d21a9533faa821a18d39bcbb8aeeb6308db603ee25afdb8978b542b3a7` | `benchmarks/otd/training/overnight-sweep-20261004/gmm_triton.py` (same) |
| `packing.py` | `08a3930937fb8aea1c51b399d9239296fdcdc5f1dcb45b78a57100049901fd00` | `benchmarks/otd/training/overnight-sweep-20261004/packing.py` (same) |
| `runtime.py` | `63a45cddc1203006cb57b7b97fb5d274932ec1654a490aa92df401ba11219e76` | `benchmarks/otd/training/final-20261007/runtime.py` (same) |
| `fit_rows.py` | `780a68a0de16c1dff79c001b85717e5f989cd8bca95be06bf0db8dd17422ea99` | `benchmarks/otd/training/final-20261007/fit_rows.py` (same) |
| `smoke_tokens.py` | `60b8021358e17f6a613a85cb227761e4317347973172c4db98f728899478c517` | `benchmarks/otd/training/final-20261007/smoke_tokens.py` (same) |
| `eval_final.py` | `6d1002000c30970214d0ea76ce750005a3e8c56f028dffe36a8cc1c90a5b0232` | `benchmarks/otd/training/final-20261007/eval_final.py` (differs) |
| `fast_eval.py` | `efef7d10967e7188b81a38f00b7012288f0c60399e3f1c4ddbebc13f6f9b26bf` | `benchmarks/otd/training/overnight-sweep-20261004/fast_eval.py` (differs) |
| `sweep_eval.py` | `e6a0ee64418cb604343b040fd849d7c00caf016ba253ac1e01a6820d4b59a14f` | `benchmarks/otd/training/overnight-sweep-20261004/sweep_eval.py` (differs) |
| `cross_model_eval.py` | `a7da027e8dd442d63793e9c07faa6bad3638cf235d4a903535f40d7e23d25107` | `benchmarks/otd/training/overnight-sweep-20261004/cross_model_eval.py` (same) |
| `score_moe.py` | `ca83818ef84774f9ad35d1d2f659b4a2dadc9a09ff1e4c19d5d1d94fff3b361d` | `research/atlas-rescore-moe-20261007/score_moe.py` (same) |
| `h200_sync.py` | `43d7a4fcacd90241ddaa21e49902a237ed693778070b00e04ec3f403d502ea0e` | `research/atlas-rescore-moe-20261007/h200_sync.py` (same) |
| `train_opt.py` | `648d789732b6af933763f2cac0756438d851c0ed4045d5657a376229e04715fe` | none (an experimental copy of the trainer that sat in the working directory; it was not run for this model) |

The run's entry point was `train_sweep.py`. `eval_final.py`, `fast_eval.py`, `sweep_eval.py` and
`cross_model_eval.py` produced its evaluation results. `score_moe.py`, `h200_sync.py`, `fit_rows.py`, `smoke_tokens.py`
and `train_opt.py` were in the same directory and are hashed because the trainer hashes every `*.py` file there.

## Supporting files (not hashed in `run.json`)

These were in the same working directory when the run was launched; they are copied from the backup of that
directory taken on 2026-10-12.

| File | SHA-256 | Note |
|---|---|---|
| `models.json` | `cd912c6dee525e8434b4ed0e936d3c1c28cb518a390def821518d06840f62cd2` | Model registry. Its `moe-final-1ep` entry equals the `model` block in `run.json`. Copy from Oct 12; entries added after Oct 10 do not affect this run. |
| `ccdshim/causal_conv1d/__init__.py` | `9bee48002ac398bb1ee0bdba8d98414fd61ab2fa272682922fa97a534edfc2d9` | Stand-in for the `causal_conv1d` package on fla's Triton kernel, needed by `--pack`. Same as `benchmarks/otd/training/overnight-sweep-20261004/ccdshim/`. |
| `mixgrid/run_abl_h200.sh` | `5c4bcb986669665d507c6d16088d93168045e760db51f548089ec159ea7890eb` | Launch wrapper. Same as `benchmarks/otd/training/loss-ablation-20261010/run_abl_h200.sh`. |
| `mixgrid/retry_lib.sh` | `a4049c398377cd999a56e7af1df9356ca03afcb654074994f9eb0783ca90556a` | Retry logic sourced by the wrapper. Same as `benchmarks/otd/training/mixgrid-20261007/retry_lib.sh`. |

## Training command

The run was launched by the loss-ablation round-1 scheduler as
`run_abl_h200.sh abl-noseg-nomix 1 <port> --skip-stage1 --segment-weight 0 --mixed-weight 0` (arm flags from
`benchmarks/otd/training/loss-ablation-20261010/arms.json`), with the W&B key on standard input. The wrapper
expands to the following, from the working directory that holds this code (`ROOT`), with `runs/moe-final-1ep/prepared-v2/`
(the training data) and `assets/moe-final-1ep/` (the base model and tokenizer) under it. Only the GPU id
(originally 1), the rendezvous port and the Python path (originally `venv/bin/python`) are generic here:

```bash
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT:$ROOT/ccdshim" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true WANDB_CONSOLE=off   # plus WANDB_API_KEY for logging
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 \
  train_sweep.py moe-final-1ep abl-noseg-nomix \
  --fraction 1.0 --stage2-epochs 1 --micro-batch 32 --lr 1e-4 --head-lr 2e-5 --schedule cosine --warmup 0.06 --seed 1 \
  --fast-moe --pack --fsdp-no-ckpt-layers 0 \
  --on-diverge rollback --snapshot-every 300 --snapshot-keep 2 --resume-every 20 --state-dir /dev/shm/woog-pangram/state-abl \
  --skip-stage1 --segment-weight 0 --mixed-weight 0
```

This argument list is copied from the first line of the run's `train.log`. The run started once and finished with
`rc=0` (no retries or resumes). The resulting `run.json` config: LoRA rank 128 (expert rank 16), alpha 32, 8-bit AdamW,
loss weights token 1, sentence 0.2, segment 0, mixed 0, document 0.1, BF16, packed attention.

Training data: the Oct 7 final mix, stage 2 merged into one 72,000-window file; `run.json` records the SHA-256 of its
`prepared-v2/manifest.json` as `0d4e4d44b303dd2717ff0a80dceb21f98706793f9cd4b144b053fd4319497273`. See
`benchmarks/otd/training/final-20261007/` (`mix_build.py`) and `benchmarks/otd/training/mixgrid-20261007/`
(`merge_epochs.py`) for how it was built. The training data itself is not in this repository.

Evaluation, after training, on the same GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT/ccdshim" HF_HUB_OFFLINE=1 FAST_EVAL=1 python -u eval_final.py abl-noseg-nomix
```

## Site scoring

The scores on the site come from `research/atlas-rescore-moe-20261007/score_atlas_fast.py` (the fast inference path
of `score_moe.py`). It imports `score_moe`, `fast_eval`, `data`, `modeling_sweep` and `adapters_short`, and ran from the
same working directory as this code, so run it with this folder and its shim on the path:
`PYTHONPATH="$ROOT:$ROOT/ccdshim" python research/atlas-rescore-moe-20261007/score_atlas_fast.py --run <run dir> --assets <base model dir> --source ... --output ... --claims ... --worker ...`
(its default `--checkpoint` is this run's `stage2-epoch0-adapters.safetensors`).

## Running this code elsewhere

`runtime.py` refuses model operations unless the code sits in one of the two directories it was written for
(`HOSTS`), and `ROOT` is this folder. To run the snapshot on another machine, either place it at one of those paths or
use the maintained versions, which are the place for any changes. Requirements on the training machine: Python 3.11,
PyTorch with CUDA, transformers 5.19.0, peft, bitsandbytes, accelerate, safetensors, triton, and `fla`
(flash-linear-attention, for the `causal_conv1d` shim).
