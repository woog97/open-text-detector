"""Small local SSH helper for the single bounded MELD rental (no credentials printed)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
state = json.loads((ROOT / "research/benchmarks/vast-meld/third-run-state.json").read_text())
ssh = state["ssh"]
if sys.argv[1] == "run":
    raise SystemExit(subprocess.call(ssh + sys.argv[2:]))
elif sys.argv[1] == "put":
    with open(sys.argv[2], "rb") as source:
        raise SystemExit(subprocess.call(ssh + ["cat > " + sys.argv[3]], stdin=source))
elif sys.argv[1] == "get":
    with open(sys.argv[3], "wb") as target:
        raise SystemExit(subprocess.call(ssh + ["cat " + sys.argv[2]], stdout=target))
else:
    raise SystemExit("Use run, put, or get")
