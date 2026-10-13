"""Reproducible Arena opening-prompt sampling. Never executes corpus instructions."""

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

B = Path(__file__).resolve().parent
D = B / "data/arena20"
SEED = "pangram4-arena20-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def read(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    )


def prepare():
    import pyarrow.parquet as pq

    if (D / "selected_prompts.jsonl").exists():
        raise ValueError("Sample already frozen; do not rebuild its frame in place")
    manifest = json.loads((D / "source_manifest.json").read_text())
    assert sha((D / "source.parquet").read_bytes()) == manifest["sha256"]
    rows = pq.read_table(
        D / "source.parquet",
        columns=[
            "question_id",
            "conversation_a",
            "conversation_b",
            "judge",
            "language",
            "tstamp",
        ],
    ).to_pylist()
    groups, exclusions = {}, []
    for row in sorted(rows, key=lambda r: r["question_id"]):
        arms = [row["conversation_a"], row["conversation_b"]]
        if any(not a or a[0]["role"] != "user" or not a[0]["content"] for a in arms):
            exclusions.append(
                {"source_id": row["question_id"], "reason": "missing_opening"}
            )
            continue
        a, b = [x[0]["content"] for x in arms]
        if a != b:
            exclusions.append(
                {"source_id": row["question_id"], "reason": "mismatched_openings"}
            )
            continue
        norm = normalize(a)
        if not norm:
            exclusions.append(
                {"source_id": row["question_id"], "reason": "empty_opening"}
            )
            continue
        pid = sha(norm.encode())
        if pid not in groups:
            groups[pid] = dict(
                prompt_id=pid,
                text=a,
                source_ids=[],
                users=[],
                languages=[],
                timestamps=[],
            )
        g = groups[pid]
        g["source_ids"].append(row["question_id"])
        g["users"].append(row["judge"])
        g["languages"].append(row["language"])
        g["timestamps"].append(row["tstamp"])
    frame = []
    for pid, g in groups.items():
        g["rank"] = sha((SEED + "\0" + pid).encode())
        g["occurrences"] = len(g["source_ids"])
        for key in ["source_ids", "users", "languages", "timestamps"]:
            g[key] = sorted(set(g[key]))
        frame.append(g)
    frame.sort(key=lambda r: (r["rank"], r["prompt_id"]))
    for i, row in enumerate(frame, 1):
        row["position"] = i
    write(D / "frame.jsonl", frame)
    write(D / "extraction_exclusions.jsonl", exclusions)
    manifest.update(
        source_rows=len(rows),
        frame_count=len(frame),
        extraction_exclusions=len(exclusions),
        frame_sha256=sha((D / "frame.jsonl").read_bytes()),
        seed=SEED,
        rubric="ARENA_20_PROTOCOL.md v1",
        extraction="Exact agreement of opening user turns; NFC/whitespace dedup; representative lowest source ID",
    )
    (D / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "source_rows",
                    "frame_count",
                    "extraction_exclusions",
                    "frame_sha256",
                ]
            }
        )
    )


def freeze():
    frame = read(D / "frame.jsonl")
    manifest = json.loads((D / "source_manifest.json").read_text())
    assert sha((D / "frame.jsonl").read_bytes()) == manifest["frame_sha256"]
    decisions = read(D / "screening.jsonl")
    assert len(decisions) <= len(frame)
    assert [r["position"] for r in decisions] == list(range(1, len(decisions) + 1))
    selected = []
    for d, row in zip(decisions, frame, strict=False):
        assert d["prompt_id"] == row["prompt_id"]
        assert (
            d["decision"] in ["eligible", "ineligible"]
            and d["rationale"]
            and d["reason"]
        )
        if d["decision"] == "eligible":
            selected.append({**row, **d})
    assert len(selected) == 20 and decisions[-1]["decision"] == "eligible"
    assert len({r["prompt_id"] for r in selected}) == 20
    payload = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in selected
    ).encode()
    path = D / "selected_prompts.jsonl"
    if path.exists():
        assert path.read_bytes() == payload, "Frozen sample differs"
    else:
        path.write_bytes(payload)
    manifest.update(
        selected_sha256=sha(payload),
        reviewed=len(decisions),
        selected=20,
        category_counts=dict(Counter(r["category"] for r in selected)),
        selected_unique_users=len({u for r in selected for u in r["users"]}),
        reviewer="Single Codex assistant; no independent second reviewer; no generated outputs available",
    )
    (D / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "selected_sha256",
                    "reviewed",
                    "category_counts",
                    "selected_unique_users",
                ]
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "freeze"])
    args = p.parse_args()
    globals()[args.command]()
