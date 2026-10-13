# Baseline mixture v1

Reusable binary ModernBERT-large LoRA baseline using existing data only. This recipe preserves existing splits and limits the eligible training pool to at most 80% of each complete source collection. Percentages count source documents/records, not repeated draws or token windows. Original paper 12k/4k/4k and GRADTEX publisher splits are preserved; they are not split again.

## Start a new run

From the project directory on this Mac:

```sh
/private/tmp/pangram-paper-hf-env/bin/python benchmarks/otd/training/baseline-mix-v1/spawn.py --name baseline-mix-01
```

Use a distinct descriptive name for each run. Add `--gpu 1` to request a particular GPU. Add `--check` to verify the frozen recipe and available GPU without creating a run or starting training. No GPU is interrupted or forcibly reclaimed.

The command launches an independent worker on the existing training Space. It runs the BF16 preparation checks before training, then trains from the pinned base model. A new run does not resume or overwrite an older one. Failures preserve their files; inspect them rather than reusing the same name.

Each run lives at `/data/workspace/baseline-mix-runs/NAME/`. Progress is in `launch-status.json` and `run/status.json`; logs in `preflight.log` and `training.log`. The automatically attached W&B URL is in `wandb-tracking.json` and `launch-status.json`. All classifier runs use `rigg-alice0/open-text-detector-classifiers`, group `text-classifiers`, with metrics-only tracking. No datasets, source code, checkpoints, configuration, machine metadata or credentials are added to the W&B payload. If tracking fails, training is not restarted: use the existing `wandb-tracking/attach.py` helper on that run directory.

## Frozen mix

| Source | Initial phase | Main phase |
|---|---:|---:|
| Human passages | 25% | 25% |
| Mirrors | 25% | 25% |
| Paper originals/replacements | 45% | 25% |
| GRADTEX | 0% | 20% |
| Generated manuscript bodies | 5% | 5% |

Shares are logical examples: 12,000 initial-phase draws and 24,000 draws in each of three main-phase epochs, sampled with replacement. GRADTEX retains document-only supervision and whole-document pooling; other sources use provenance-labeled windows. Source percentages do not equal token or class percentages. This release preserves the existing trainer's objective, initialization and optimizer settings; it does not claim a new hyperparameter search.

## Splits and evaluation

The durable Space recipe is `/data/workspace/baseline-mix-v1/recipe.json`. It pins checksums of prepared data, trainer code, configuration, model identity, worker, and tracker. Each launch checks these before cloning an isolated run. The launcher uses the existing Space model/tokenizer assets and downloads none locally.

`split-report.json` records full-source denominators and eligible training counts. `prepared-v2/split-assignments.json` preserves existing split decisions and removes explicitly reserved manuscript source families from every training source. Whole connected groups are removed together. Only affected training draws are replaced, preserving source shares and the rest of the original schedule.

The 120 explicitly reserved manuscripts and their related families cannot enter new runs. Original validation/test source files remain in place. New removals are also retained as `prepared-v2/reserved-SOURCE.jsonl.gz` where present. The reservation lists all evaluation manuscripts, including those already outside the baseline pool. Paper and GRADTEX original validation/test splits remain authoritative, even though the baseline's assignment file inventories only their imported train records.

Outside-training counts include exclusions and duplicate removal, not just evaluation examples. Do not score excluded records automatically. Previously trained checkpoints may have seen newly reserved manuscripts: the split report records overlap with the old schedule; those manuscripts are not clean tests of such checkpoints. This recipe does not alter an already running or completed baseline.

The one-command launch is for the current validated ModernBERT trainer. Other backbone trainers can consume the same source pools and split assignments but must build and validate their own token windows.
