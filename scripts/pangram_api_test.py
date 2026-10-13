"""Small Pangram API checks. Default: model discovery only; never print the key."""

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://text.external-api.pangram.com"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def load_key():
    path = ROOT / ".env.pangram"
    for line in path.read_text().splitlines():
        if line.startswith("PANGRAM_API_KEY="):
            key = line.split("=", 1)[1].strip().strip("\"'")
            if key and not any(c.isspace() for c in key):
                return key
    raise ValueError("Fill in PANGRAM_API_KEY in .env.pangram first.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-file", type=Path, help="Optional paid scan, limited to 100 whitespace-separated words.")
    parser.add_argument("--model", help="Required for a paid scan; use a selector from model discovery.")
    args = parser.parse_args()
    if args.scan_file and not args.model:
        parser.error("--scan-file requires --model")
    text = args.scan_file.read_text() if args.scan_file else None
    if text is not None and not 1 <= len(text.split()) <= 100:
        parser.error("A test input must contain 1–100 words.")
    key = load_key()
    opener = build_opener(NoRedirect())

    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(BASE + path, data=data, headers={
            "x-api-key": key, "Content-Type": "application/json",
        })
        try:
            with opener.open(req, timeout=30) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            raise RuntimeError(f"Pangram returned HTTP {exc.code}; response body suppressed.") from None
        except (URLError, TimeoutError):
            raise RuntimeError("Network request failed. No automatic submission retry was made.") from None

    def show(value):
        print(json.dumps(value, indent=2).replace(key, "[REDACTED]"), flush=True)

    catalog = request("/models")
    show(catalog)
    if text is None:
        return
    if args.model not in catalog.get("models", []):
        raise ValueError("Selected model is not in the current model catalog.")
    submitted = request("/task", {"text": text, "model": args.model, "public_dashboard_link": False})
    task_id = submitted["task_id"]
    show({"task_id": task_id})
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        result = request("/task/" + quote(task_id, safe=""))
        if result.get("stage") in ("STAGE_SUCCESS", "STAGE_FAILED"):
            show(result)
            return 0 if result["stage"] == "STAGE_SUCCESS" else 1
        time.sleep(2)
    raise RuntimeError("Polling timed out. Retain the task ID; do not resubmit the scan.")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("Test failed; detailed exception suppressed to protect credentials.", file=sys.stderr)
        sys.exit(1)
