---
title: Open Text Detector Training
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Open Text Detector Training

Private shared JupyterLab workspace for detector experiments. This is an
environment scaffold, not a trained detector or a complete training pipeline.

## Provisioning

- Space: `open-text-detector/training` (private, initially CPU Basic).
- Collaborators: `woog` (organization owner), `eac123` (write invitation sent).
- Set `JUPYTER_TOKEN` as a Space Secret before starting; startup fails without it.
- Both collaborators use their own HF accounts and the notebook login token.
- Private bucket `open-text-detector/training-storage` is mounted read/write at `/data`
  for durable notebooks/checkpoints.
  Without it, `/home/user/workspace` is temporary and is lost on container replacement.
- GPU access must be enabled separately by the HF contact. Do not activate a paid
  hardware tier without confirmed coverage.

Open the Space, enter the notebook token, then use JupyterLab's terminal or notebooks.
Run `python /home/user/app/check_environment.py` to inspect the environment.
Run `python /home/user/app/check_environment.py --require-gpu` after GPU provisioning.
This tests a tiny forward/backward computation, not detector training or accuracy.

The image includes the project's current CUDA runner's Torch/Transformers versions,
plus notebook and training utilities. Large datasets, model weights, local history,
and credentials are not included. Add only the selected training code and data when
the experiment is defined. Save long-running experiment checkpoints to durable storage.

The notebook server listens on port 7860. Keep the Space private and token protection
enabled. Each Space has one shared filesystem/process environment; concurrent users
must coordinate GPU jobs.

## Build recovery — October 3, 2026

Spaces development mode injects an OpenVSCode installation step that calls `wget`.
Keep `wget` in the Dockerfile's system packages; without it, the build fails with
`/bin/sh: 1: wget: not found` after Python dependencies install successfully.
Fix deployed in Space commit `b8151f58f2249a65fb39340c85e0b7dc4371af89`.
After rebuilding, authenticated remote execution succeeded, `/data/workspace`
was readable and writable, and all four A100 80 GB GPUs were visible and idle.
This is a recovery snapshot; recheck live job state before starting work.
