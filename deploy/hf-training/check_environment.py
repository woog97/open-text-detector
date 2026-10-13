"""Small CPU/GPU forward/backward check; no model or dataset download."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    import torch
    import transformers

    available = torch.cuda.is_available()
    if args.require_gpu and not available:
        raise SystemExit("GPU unavailable. Ask the HF contact to enable the allocation.")
    device = "cuda" if available else "cpu"
    model = torch.nn.Linear(8, 2).to(device)
    loss = model(torch.ones(4, 8, device=device)).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps({
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": device, "gpu_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if available else None,
        "forward_backward": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
