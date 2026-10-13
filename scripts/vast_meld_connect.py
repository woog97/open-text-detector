"""Enforce the user's three-minute boot deadline before beginning SSH setup."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
p = Path(sys.argv[1])
s = json.loads(p.read_text())
key = (Path.home() / ".config/vastai/vast_api_key").read_text().strip()
url = f'https://console.vast.ai/api/v0/instances/{s["instance_id"]}/'
headers = {"Authorization": "Bearer " + key}
while time.time() < s["started_at"] + 180:
    x = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20)).get("instances", {})
    if x.get("actual_status") == "running":
        direct = (x.get("ports") or {}).get("22/tcp")
        host = x["public_ipaddr"] if direct else x["ssh_host"]
        port = direct[0]["HostPort"] if direct else x["ssh_port"]
        s["ssh"] = ["ssh", "-i", str(Path.home()/".ssh/id_ed25519"), "-o", "IdentitiesOnly=yes",
                    "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "UserKnownHostsFile=" + str(ROOT/"research/benchmarks/vast-meld/known_hosts"),
                    "-p", str(port), "root@" + host]
        p.write_text(json.dumps(s, indent=2))
        print("RUNNING", host, port, "startup_seconds", round(time.time() - s["started_at"]), flush=True)
        subprocess.run(s["ssh"] + ["mkdir -p /workspace/pangram; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"], check=True)
        break
    print("Startup", x.get("actual_status"), round(time.time() - s["started_at"]), flush=True)
    time.sleep(min(10, max(0, s["started_at"] + 180 - time.time())))
else:
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, method="DELETE", headers=headers), timeout=20))
    assert d.get("success"), d
    p.with_suffix(".destroyed").write_text(json.dumps(d))
    print("Destroyed after 3-minute startup limit", flush=True)
