"""One-time source discovery; no model or website access."""

import concurrent.futures
import json
from pathlib import Path

import httpx

BASE = Path(__file__).resolve().parent
urls = {
    "epoch-tree": "https://api.github.com/repos/jaeholee-brown/ai-text-detectors/git/trees/main?recursive=1",
    "saha-tree": "https://api.github.com/repos/FLAIR-IISc/ai-in-peer-review/git/trees/main?recursive=1",
    "sem-metadata": "https://huggingface.co/api/datasets/Sem-Detect/ML_Conferences-Peer-Reviews",
    "meld-tree": "https://huggingface.co/api/datasets/anon-review-meld-2026/meld-eval/tree/4c80f0fdc003854e5b6bfda90f7ac41ffb12e232",
    "opai-tree": "https://huggingface.co/api/datasets/OpAI-Bench1/OpAI-Bench/tree/4fb59011bcab8c62863b7083fe0f0a3345555558/default/test",
}


def fetch(k, u):
    r = httpx.get(u, timeout=60, follow_redirects=True)
    r.raise_for_status()
    d = r.json()
    (BASE / "sources" / f"{k}.json").write_text(json.dumps(d, indent=2))
    if isinstance(d, list):
        print(k, [(x["path"], x.get("size")) for x in d])
    else:
        print(
            k,
            d.get("sha"),
            [
                (x.get("path", x.get("rfilename")), x.get("size"))
                for x in d.get("tree", d.get("siblings", []))
                if x.get("type") != "tree"
            ][:35],
        )


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    list(ex.map(lambda kv: fetch(*kv), urls.items()))
