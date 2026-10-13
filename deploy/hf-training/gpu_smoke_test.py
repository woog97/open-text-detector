"""Bounded synthetic training + checkpoint/resume test; no downloads or real data."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Local validation only; default requires CUDA.")
    parser.add_argument("--output-dir", default="/data/workspace/smoke-tests")
    args = parser.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: this test requires a working GPU.")
    device = torch.device("cpu" if args.cpu else "cuda:0")
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(args.output_dir) / run_id
    output.mkdir(parents=True, exist_ok=False)

    # A learnable synthetic regression task, with independent validation inputs.
    teacher = torch.randn(64, 8, device=device) / 8
    x = torch.randn(2048, 64, device=device)
    y = x @ teacher
    vx = torch.randn(512, 64, device=device)
    vy = vx @ teacher

    def make_model():
        return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 8)).to(device)

    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    with torch.no_grad():
        initial = nn.functional.mse_loss(model(vx), vy).item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.monotonic()
    for step in range(100):
        if time.monotonic() - started > 60:
            raise TimeoutError("Training exceeded the 60-second test budget.")
        indices = torch.randint(len(x), (256,), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x[indices]), y[indices])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    with torch.no_grad():
        expected = model(vx)
        final = nn.functional.mse_loss(expected, vy).item()
    if final >= initial * 0.25:
        raise RuntimeError(f"Learning check failed: validation loss {initial} -> {final}.")

    checkpoint = output / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 100}, checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    restored = make_model()
    restored.load_state_dict(state["model"])
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(state["optimizer"])
    with torch.no_grad():
        torch.testing.assert_close(restored(vx), expected, rtol=0, atol=0)
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_loss = nn.functional.mse_loss(restored(x[:256]), y[:256])
    resumed_loss.backward()
    resumed_optimizer.step()
    if not all(torch.isfinite(p).all().item() for p in restored.parameters()):
        raise RuntimeError("Non-finite parameters after checkpoint resume.")

    result = {
        "status": "passed", "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "training_steps": 100, "resumed_steps": 1,
        "initial_validation_loss": initial, "final_validation_loss": final,
        "training_seconds": elapsed, "checkpoint_reload_exact": True,
        "checkpoint": str(checkpoint),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("GPU_SMOKE_TEST_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
