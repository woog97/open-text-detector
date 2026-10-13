"""Run/resume authorized Arena-100 generation. Secrets go only to OpenRouter."""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from arena20 import B, read, sha, write
from arena_generate import audit_signals

D = B / "data/arena100"
R = B / "runs/arena100"
REQUEST_TIMEOUT = 180


def now():
    return datetime.now(timezone.utc).isoformat()


def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n")


def append(p, x):
    with p.open("a") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def credentials():
    return next(
        s.split("=", 1)[1].strip().strip("\"'")
        for s in (B / ".env.secrets").read_text().split("\n")
        if s.startswith("OPENROUTER_API_KEY=")
    )


async def fetch(path, body=None, auth=True):
    args = [
        "curl",
        "-sS",
        "--connect-timeout",
        "15",
        "--max-time",
        str(REQUEST_TIMEOUT),
        "--config",
        "-",
        "-w",
        "\n%{http_code}",
        "https://openrouter.ai/api/v1/" + path,
    ]
    config = 'header = "Content-Type: application/json"\n'
    if auth:
        config += 'header = "Authorization: Bearer ' + credentials() + '"\n'
    if body is not None:
        args += ["-X", "POST"]
        # curl config quoting; body and credential remain off the command line.
        data = json.dumps(body, ensure_ascii=False)
        config += (
            'data-binary = "'
            + data.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            + '"\n'
        )
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate(config.encode())
    if proc.returncode:
        return 0, {"error": {"type": "curl_transport", "exit_code": proc.returncode}}
    payload, code = out.decode().rsplit("\n", 1)
    try:
        j = json.loads(payload)
    except ValueError:
        j = {"error": {"type": "non_json"}}
    return int(code), j


def body(m, p, manifest, batch=False):
    x = {
        "messages": [
            {"role": "system", "content": manifest["system"]},
            {"role": "user", "content": p["text"]},
        ],
        **m["settings"],
    }
    if not batch:
        x.update(model=m["id"], provider=m["provider"])
    return x


def output(m, p, request, attempts, raw, error=None):
    return {
        "model_id": m["id"],
        "prompt_id": p["prompt_id"],
        "request": request,
        "request_sha256": sha(json.dumps(request, sort_keys=True).encode()),
        "attempts": attempts,
        "success": raw is not None,
        "raw": raw,
        "error": error,
        "audit": audit_signals(raw) if raw else None,
    }


def effective_status(status, response):
    # OpenRouter can return an inline provider error after HTTP 200 headers.
    error = response.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    return code if status == 200 and isinstance(code, int) else status


async def main(a):
    global REQUEST_TIMEOUT
    REQUEST_TIMEOUT = a.request_timeout
    R.mkdir(parents=True, exist_ok=True)
    mf = D / "generation_manifest.json"
    manifest = json.loads(mf.read_text())
    prompts = read(D / "selected_prompts.jsonl")
    assert (
        len(prompts) == 100
        and sha((D / "selected_prompts.jsonl").read_bytes())
        == manifest["selected_prompts_sha256"]
    )
    assert len(manifest["models"]) == manifest.get("planned_models", 12)
    if a.dry_run:
        print(f"Validated 100 prompts × {len(manifest['models'])} models.")
        return
    dest = R / "generation_manifest.json"
    if dest.exists():
        assert dest.read_bytes() == mf.read_bytes()
    else:
        dest.write_bytes(mf.read_bytes())
    for p in [
        B / "arena100_generate.py",
        B / "arena100_prepare.py",
        B / "ARENA_100_PROTOCOL.md",
        D / "selected_prompts.jsonl",
        D / "screening.jsonl",
        D / "source_manifest.json",
    ]:
        dest = R / p.name
        if not dest.exists():
            dest.write_bytes(p.read_bytes())
    status, cat = await fetch("models", auth=False)
    assert status == 200
    lookup = {m["id"]: m for m in cat["data"]}
    for m in manifest["models"]:
        assert lookup[m["catalog_id"]]["canonical_slug"] == m["canonical_slug"]
    route_overrides = {}
    if a.route_overrides:
        assert a.retry_failed, "Route changes apply only to explicitly retried failures"
        route_overrides = json.loads(a.route_overrides.read_text())["overrides"]
        by_id = {m["id"]: m for m in manifest["models"]}
        for mid, override in route_overrides.items():
            assert set(override) == {"model", "provider"}, (
                "Only provider routing may change"
            )
            m = by_id[mid]
            assert lookup[override["model"]]["canonical_slug"] == m["canonical_slug"], (
                "Cross-model fallback forbidden"
            )
            for key in ["prompt", "completion", "request"]:
                assert (
                    override["provider"]["max_price"][key]
                    <= m["provider"]["max_price"][key]
                ), "Cannot raise frozen price cap"
    save(R / "catalog_at_execution.json", cat)
    bm = next((m for m in manifest["models"] if m["mode"] == "batch"), None)
    batch_path = R / "batch_submission.json"
    if bm is not None and not batch_path.exists():
        assert not (R / "batch_submission_started.json").exists(), (
            "Uncertain batch submission: reconcile before retry"
        )
        batch = {
            "endpoint": "/v1/chat/completions",
            "model": bm["id"],
            "provider": manifest["batch_provider"],
            "completion_window": "24h",
            "requests": [
                {"custom_id": p["prompt_id"], "body": body(bm, p, manifest, True)}
                for p in prompts
            ],
        }
        save(R / "batch_request.json", batch)
        save(
            R / "batch_submission_started.json",
            {"at": now(), "sha256": sha(json.dumps(batch).encode())},
        )
        status, j = await fetch("batches", batch)
        save(batch_path, {"http_status": status, "response": j})
        print("Batch submission", status, j.get("id"), j.get("status"), flush=True)
        if status not in [200, 202]:
            print("Batch error", j, flush=True)
    if a.batch_only:
        return
    all_paths = [R / "responses.jsonl"] + sorted(R.glob("responses_resume_*.jsonl"))
    prior_records = [x for path in all_paths if path.exists() for x in read(path)]
    latest = {(x["model_id"], x["prompt_id"]): x for x in prior_records}
    existing = list(latest.values())
    done = set(latest)
    attempts_path = R / "attempts.jsonl"
    outpath = R / "responses.jsonl"
    if a.resume_blocked or a.retry_failed:
        number = len(list(R.glob("responses_resume_*.jsonl"))) + 1
        outpath = R / f"responses_resume_{number:03d}.jsonl"
        attempts_path = R / f"attempts_resume_{number:03d}.jsonl"
        # Only explicitly rejected calls may be resubmitted after the user resolves billing.
        for k, x in latest.items():
            if not x["success"]:
                last = x["attempts"][-1]
                last_status = last.get("effective_status", last.get("status"))
                message = str((last.get("response") or {}).get("error") or "")
                if (
                    a.retry_failed
                    or last_status == 429
                    or (last_status == 403 and "Key limit exceeded" in message)
                ):
                    done.remove(k)
        outpath.touch(exist_ok=False)
        outpath.with_suffix(".runner.py").write_bytes(Path(__file__).read_bytes())
        save(
            outpath.with_suffix(".settings.json"),
            {
                "at": now(),
                "retry_all_failures": a.retry_failed,
                "concurrency": a.concurrency,
                "request_timeout": REQUEST_TIMEOUT,
                "route_overrides": route_overrides,
                "authorization": "User requested retrying failed cells to complete the selected model/prompt grid"
                if a.retry_failed
                else "Resume after key allowance update",
                "uncertain_prior_transport": [
                    list(k)
                    for k, x in latest.items()
                    if not x["success"] and x["attempts"][-1].get("status") == 0
                ],
                "recovery_note": "Prior curl timeouts contain no generation ID or saved response content; lookup by generation ID cannot recover those responses. Retried only under explicit user authorization; possible prior charges remain unverified."
                if a.retry_failed
                else None,
            },
        )
    for apath in [R / "attempts.jsonl"] + sorted(R.glob("attempts_resume_*.jsonl")):
        if not apath.exists():
            continue
        starts = read(apath)
        uncertain = set()
        for x in starts:
            if x["event"] != "start":
                continue
            key = (x["model_id"], x["prompt_id"])
            recorded = latest.get(key, {}).get("attempts", [])
            # An older billing rejection cannot reconcile a newer interrupted call.
            if not recorded or max(a.get("at", "") for a in recorded) < x["at"]:
                uncertain.add(key)
        assert not uncertain, "Uncertain sync completions require reconciliation"
    blocked = asyncio.Event()
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    count = [len(existing)]

    async def run(m, p):
        async with sem:
            if blocked.is_set():
                return
            req = body(m, p, manifest)
            if m["id"] in route_overrides:
                req.update(route_overrides[m["id"]])
            attempts = []
            raw = None
            for n in range(1, 4):
                identity = {"model_id": m["id"], "prompt_id": p["prompt_id"]}
                stamp = now()
                start = time.monotonic()
                append(
                    attempts_path,
                    {
                        **identity,
                        "event": "start",
                        "attempt": n,
                        "at": stamp,
                        "request_sha256": sha(json.dumps(req, sort_keys=True).encode()),
                    },
                )
                status, j = await fetch("chat/completions", req)
                retry_status = effective_status(status, j)
                record = {
                    "attempt": n,
                    "at": stamp,
                    "status": status,
                    "effective_status": retry_status,
                    "seconds": time.monotonic() - start,
                    "response": j,
                }
                attempts.append(record)
                append(attempts_path, {**identity, "event": "finish", **record})
                if status == 200 and j.get("choices") and not j.get("error"):
                    raw = j
                    break
                if retry_status == 403 and "Key limit exceeded" in str(j.get("error")):
                    blocked.set()
                    print(
                        "Paused new calls: API key spending limit reached", flush=True
                    )
                    break
                # Never automatically retry uncertain transport outcomes.
                if n == 3 or not (retry_status == 429 or 500 <= retry_status < 600):
                    break
                await asyncio.sleep(n * 5)
            async with lock:
                append(
                    outpath,
                    output(
                        m,
                        p,
                        req,
                        attempts,
                        raw,
                        None if raw else "API/transport failure",
                    ),
                )
                count[0] += 1
                if count[0] % 10 == 0:
                    print(
                        json.dumps({"completed_sync": count[0], "at": now()}),
                        flush=True,
                    )

    await asyncio.gather(
        *(
            run(m, p)
            for p in prompts
            for m in manifest["models"]
            if m["mode"] != "batch" and (m["id"], p["prompt_id"]) not in done
        )
    )
    print(
        "Synchronous grid paused on key limit"
        if blocked.is_set()
        else "Synchronous grid finished",
        flush=True,
    )


async def poll():
    mfest = json.loads((D / "generation_manifest.json").read_text())
    m = next(m for m in mfest["models"] if m["mode"] == "batch")
    sub = json.loads((R / "batch_submission.json").read_text())
    j = sub["response"]
    if j.get("id"):
        status, j = await fetch("batches/" + j["id"])
        assert status == 200
        save(R / "batch_latest.json", j)
        print(
            json.dumps(
                {
                    k: j.get(k)
                    for k in ["id", "status", "request_counts", "usage", "error"]
                }
            ),
            flush=True,
        )
    if j.get("status") not in ["completed", "failed", "expired", "cancelled"] and sub[
        "http_status"
    ] in [200, 202]:
        return
    path = R / "responses_batch.jsonl"
    if path.exists():
        return
    results = {x["custom_id"]: x for x in j.get("results") or []}
    rows = []
    for p in read(D / "selected_prompts.jsonl"):
        x = results.get(p["prompt_id"], {})
        res = x.get("response") or {}
        raw = res.get("body")
        success = (
            res.get("status_code") == 200
            and isinstance(raw, dict)
            and bool(raw.get("choices"))
            and not raw.get("error")
        )
        rows.append(
            output(
                m,
                p,
                body(m, p, mfest, True),
                [
                    {
                        "batch_id": j.get("id"),
                        "status": res.get("status_code"),
                        "error": x.get("error") or j.get("error"),
                    }
                ],
                raw if success else None,
                None if success else "Batch failure/missing result",
            )
        )
    write(path, rows)
    print("Batch cells preserved", len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-only", action="store_true")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--resume-blocked", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--request-timeout", type=int, default=180)
    ap.add_argument("--route-overrides", type=Path)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument(
        "--experiment",
        choices=["arena100", "arena100-cheap", "arena100-skipped", "arena100-kimi"],
        default="arena100",
    )
    a = ap.parse_args()
    D = B / "data" / a.experiment
    R = B / "runs" / a.experiment
    asyncio.run(poll() if a.poll else main(a))
