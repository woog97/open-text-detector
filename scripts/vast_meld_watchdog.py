"""Independent deadline enforcement for this explicitly authorized temporary rental.

Runs locally; Vast credentials never leave the user's machine. Only the instance
recorded in this run's state file can be destroyed. Retrieval failure must not
extend the rental deadline.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

statefile = Path(sys.argv[1]).resolve()
keyfile = Path.home() / ".config/vastai/vast_api_key"


def api(method, instance):
    req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{instance}/",
        method=method, headers={"Authorization": "Bearer " + keyfile.read_text().strip()})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


while not statefile.exists():
    time.sleep(2)
state = json.loads(statefile.read_text())
instance = int(state["instance_id"])
assert state["label"].startswith("meld-5080-one-hour-")
log(f"Watching authorized instance {instance}, destroy deadline {state['destroy_at']}")
while time.time() < state["retrieve_at"]:
    if statefile.with_suffix(".destroyed").exists():
        sys.exit(0)
    time.sleep(min(10, max(0, state["retrieve_at"] - time.time())))
state = json.loads(statefile.read_text())
if state.get("ssh"):
    try:
        with statefile.with_name("watchdog-results.tar.gz").open("wb") as target:
            subprocess.run(state["ssh"] + ["tar -C /workspace/pangram -czf - " + state.get("results_path", "research/benchmarks/meld-cuda")],
                           stdout=target, stderr=subprocess.PIPE, timeout=90, check=True)
        log("Final result archive retrieved")
    except Exception as exc:
        log(f"Retrieval failed: {type(exc).__name__}; deadline remains enforced")
while time.time() < state["destroy_at"]:
    if statefile.with_suffix(".destroyed").exists():
        sys.exit(0)
    time.sleep(min(5, max(0, state["destroy_at"] - time.time())))
for attempt in range(12):
    try:
        result = api("DELETE", instance)
        log(f"Destroy response: {result}")
        if result.get("success"):
            statefile.with_suffix(".destroyed").write_text(json.dumps(result))
            break
    except Exception as exc:
        log(f"Destroy attempt {attempt + 1}: {type(exc).__name__}")
    time.sleep(5)
else:
    log("URGENT: Could not confirm destruction; inspect Vast account immediately")
    sys.exit(1)
