"""Resume the frozen Arena-20 OpenRouter grid, preserving first responses and failures."""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from arena20 import B, D, read, sha

R = B / "runs/arena20"


def request_body(manifest, model, prompt):
    return {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": manifest["system"]},
            {"role": "user", "content": prompt["text"]},
        ],
        "provider": manifest["provider"],
        **model["settings"],
    }


def audit_signals(raw):
    choices = raw.get("choices") or []
    choice = choices[0] if choices else {}
    text = choice.get("message", {}).get("content") or ""
    if not isinstance(text, str):
        text = ""
    finish = choice.get("finish_reason")
    return {
        "text": text,
        "words": len(text.split()),
        "finish_reason": finish,
        "length_ok": len(text.split()) >= 50,
        "not_truncated": finish == "stop",
        "input_output_token_condition": "unverified: provider prompt usage includes system/chat formatting; native user-only tokenizer unavailable",
        "manual_content_review": "pending",
    }


async def main(args):
    global R
    R = args.output
    R.mkdir(parents=True, exist_ok=True)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    prompts = read(D / "selected_prompts.jsonl")
    assert (
        sha((D / "selected_prompts.jsonl").read_bytes())
        == manifest["selected_prompts_sha256"]
    )
    assert len(prompts) == 20 and len(manifest["models"]) > 0
    expected_cells = 20 * len(manifest["models"])
    assert len({p["prompt_id"] for p in prompts}) == 20
    assert float(manifest["provider"]["max_price"]["completion"]) < 2
    frozen = R / "generation_manifest.json"
    if frozen.exists():
        assert frozen.read_bytes() == manifest_bytes, (
            "Do not change the experiment while resuming"
        )
    else:
        frozen.write_bytes(manifest_bytes)
    for name in ["arena_generate.py", "arena20.py", "ARENA_20_PROTOCOL.md"]:
        dest = R / name
        if not dest.exists():
            dest.write_bytes((B / name).read_bytes())
    secrets = {}
    for line in (B / ".env.secrets").read_text().split("\n"):
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            secrets[key.strip()] = value.strip().strip("\"'")
    key = secrets.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env.secrets")
    output = R / "responses.jsonl"
    done = (
        {(r["model_id"], r["prompt_id"]) for r in read(output)}
        if output.exists()
        else set()
    )
    # A start without a terminal record has uncertain billing/output; never silently replay it.
    started_path = R / "attempts.jsonl"
    prior = read(started_path) if started_path.exists() else []
    uncertain = {
        (r["model_id"], r["prompt_id"]) for r in prior if r["event"] == "start"
    } - done
    if uncertain:
        raise ValueError(
            f"{len(uncertain)} unfinished cells require reconciliation; refusing duplicate spend"
        )
    calls = [
        (m, p)
        for p in prompts
        for m in manifest["models"]
        if (m["id"], p["prompt_id"]) not in done
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "remaining_cells": len(calls),
                    "sample_sha256": manifest["selected_prompts_sha256"],
                    "model_count": len(manifest["models"]),
                }
            )
        )
        return
    # Compare current routes to frozen catalog identities before any paid request.
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180, connect=30), follow_redirects=False
    ) as client:
        catalog_response = await client.get("https://openrouter.ai/api/v1/models")
        catalog_response.raise_for_status()
        catalog = {m["id"]: m for m in catalog_response.json()["data"]}
        for m in manifest["models"]:
            assert (
                catalog.get(m["id"], {}).get("canonical_slug") == m["canonical_slug"]
            ), f"Model changed: {m['id']}"
        (R / "catalog_at_execution.json").write_text(
            json.dumps(catalog_response.json()) + "\n"
        )
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        free_lock = asyncio.Lock()
        last_free = [0.0]
        counts = {
            "completed": len(done),
            "successful": 0,
            "failed": 0,
            "reported_cost_usd": 0.0,
        }

        def append(path, record):
            with path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

        async def run(model, prompt):
            async with sem:
                body = request_body(manifest, model, prompt)
                identity = {"model_id": model["id"], "prompt_id": prompt["prompt_id"]}
                attempts, raw, error = [], None, None
                for number in range(1, 4):
                    if model["id"].endswith(":free"):
                        async with free_lock:
                            await asyncio.sleep(
                                max(0, 3.2 - (time.monotonic() - last_free[0]))
                            )
                            last_free[0] = time.monotonic()
                    start = time.monotonic()
                    stamp = datetime.now(timezone.utc).isoformat()
                    async with lock:
                        append(
                            started_path,
                            {
                                **identity,
                                "event": "start",
                                "attempt": number,
                                "at": stamp,
                                "request_sha256": sha(
                                    json.dumps(body, sort_keys=True).encode()
                                ),
                            },
                        )
                    retry = False
                    try:
                        response = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": "Bearer " + key},
                            json=body,
                        )
                        try:
                            payload = response.json()
                        except ValueError:
                            payload = {
                                "error": {
                                    "message": "Non-JSON response",
                                    "status": response.status_code,
                                }
                            }
                        record = {
                            "attempt": number,
                            "at": stamp,
                            "status": response.status_code,
                            "seconds": time.monotonic() - start,
                            "response": payload,
                        }
                        if (
                            response.status_code == 200
                            and payload.get("choices")
                            and not payload.get("error")
                        ):
                            raw, error = payload, None
                        else:
                            error = "HTTP/API failure"
                            retry = (
                                response.status_code == 429
                                or 500 <= response.status_code < 600
                            )
                    except httpx.TransportError as exc:
                        # Exception type only: never write authenticated request headers.
                        error = type(exc).__name__
                        record = {
                            "attempt": number,
                            "at": stamp,
                            "error": error,
                            "seconds": time.monotonic() - start,
                        }
                        retry = True
                    attempts.append(record)
                    async with lock:
                        append(started_path, {**identity, "event": "finish", **record})
                    if raw is not None or not retry or number == 3:
                        break
                    await asyncio.sleep(5 * number)
                out = {
                    **identity,
                    "request": body,
                    "request_sha256": sha(json.dumps(body, sort_keys=True).encode()),
                    "attempts": attempts,
                    "success": raw is not None,
                    "error": error,
                    "raw": raw,
                    "audit": audit_signals(raw) if raw else None,
                }
                async with lock:
                    append(output, out)
                    counts["completed"] += 1
                    counts["successful" if raw else "failed"] += 1
                    if raw:
                        counts["reported_cost_usd"] += float(
                            (raw.get("usage") or {}).get("cost") or 0
                        )
                    print(json.dumps(counts), flush=True)

        await asyncio.gather(*(run(m, p) for m, p in calls))
    final = read(output)
    assert (
        len(final) == expected_cells
        and len({(r["model_id"], r["prompt_id"]) for r in final}) == expected_cells
    )
    print(f"Frozen {expected_cells}-cell generation grid complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=D / "generation_manifest.json")
    parser.add_argument("--output", type=Path, default=R)
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(parser.parse_args()))
