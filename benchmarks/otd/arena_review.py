"""Local response review packets and preparation for baseline detectors. No network."""

import argparse
import json
from collections import Counter

from arena20 import B, read, sha, write
from prepare import record, validate

R = B / "runs/arena20"
INPUTS = [R / "responses.jsonl", B / "runs/arena20-replacements/responses.jsonl"]


def collect():
    return [r for p in INPUTS if p.exists() for r in read(p)]


def packet(limit):
    reviews = (
        read(R / "content_reviews.jsonl")
        if (R / "content_reviews.jsonl").exists()
        else []
    )
    done = {(r["model_id"], r["prompt_id"]) for r in reviews}
    candidates = [
        r
        for r in collect()
        if r["success"]
        and r["audit"]["length_ok"]
        and r["audit"]["not_truncated"]
        and (r["model_id"], r["prompt_id"]) not in done
    ][:limit]
    if (R / "review_packet.jsonl").exists():
        raise ValueError("Existing packet must be adjudicated before creating another")
    write(R / "review_packet.jsonl", candidates)
    for i, r in enumerate(candidates, 1):
        text = r["audit"]["text"]
        lines = text.splitlines()
        # Label-free excerpts; reviewers can request a full cell by packet index.
        print(
            json.dumps(
                {
                    "i": i,
                    "words": r["audit"]["words"],
                    "start": text[:600],
                    "end": text[-160:] if len(text) > 760 else "",
                    "format": Counter(
                        "list"
                        if x.strip().startswith(
                            ("- ", "* ", "1.", "2.", "3.", "|", "```")
                        )
                        else "other"
                        for x in lines
                        if x.strip()
                    ),
                },
                ensure_ascii=False,
            )
        )


def finalize():
    rows = collect()
    reviews = read(R / "content_reviews.jsonl")
    review = {(r["model_id"], r["prompt_id"]): r for r in reviews}
    prepared = []
    audits = []
    for r in rows:
        k = (r["model_id"], r["prompt_id"])
        a = r.get("audit") or {}
        text = a.get("text", "")
        decision = {
            "model_id": k[0],
            "prompt_id": k[1],
            "response_sha256": sha(text.encode()),
            "native_token_gate": "unverified",
            "words": a.get("words", 0),
        }
        if not r["success"]:
            decision.update(eligible=False, reason="api_failure")
        elif not a["not_truncated"]:
            decision.update(eligible=False, reason="non_stop_finish")
        elif not a["length_ok"]:
            decision.update(eligible=False, reason="below_50_words")
        else:
            assert k in review, f"Unreviewed cell {k}"
            assert review[k]["response_sha256"] == decision["response_sha256"]
            decision.update(
                eligible=review[k]["eligible"],
                reason=review[k]["reason"],
                review_method=review[k]["review_method"],
            )
        audits.append(decision)
        if decision["eligible"]:
            prepared.append(
                record(
                    "arena20",
                    sha((k[0] + "\0" + k[1]).encode()),
                    text,
                    "ai",
                    generator=k[0],
                    cohort=k[0],
                    group_id=k[1],
                    prompt_id=k[1],
                    provenance="Fresh OpenRouter generation, frozen Arena-20 prompt; scope screened locally; native length gate unverified",
                    task="ai_detection",
                    response_sha256=decision["response_sha256"],
                )
            )
    write(R / "response_audit.jsonl", audits)
    validate(prepared)
    write(B / "data/prepared/arena20.jsonl", prepared)
    print(
        json.dumps(
            {
                "cells": len(rows),
                "eligible_exploratory": len(prepared),
                "exclusions": Counter(r["reason"] for r in audits if not r["eligible"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["packet", "finalize"])
    parser.add_argument("--limit", type=int, default=40)
    a = parser.parse_args()
    packet(a.limit) if a.command == "packet" else finalize()
