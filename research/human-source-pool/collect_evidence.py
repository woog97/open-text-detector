"""Snapshot small public source metadata, never model assets or corpus shards."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "evidence"
HF_IDS = ["common-pile/" + x for x in (
    "cccc", "peS2o", "pubmed", "arxiv_papers", "project_gutenberg",
    "stackexchange", "wikimedia", "pressbooks", "libretexts", "news",
    "foodista", "public_domain_review", "usgpo", "ubuntu_irc")]
URLS = {x.replace("/", "--") + ".json": "https://huggingface.co/api/datasets/" + x
        for x in HF_IDS}
URLS.update({
    "asap2-repository.json": "https://api.github.com/repos/scrosseye/ASAP_2.0/commits/main",
    "asap2-readme.txt": "https://raw.githubusercontent.com/scrosseye/ASAP_2.0/main/README.md",
    "persuade2-readme.txt": "https://raw.githubusercontent.com/scrosseye/persuade_corpus_2.0/main/README.md",
    "ellipse-readme.txt": "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/README.md",
    "writingprompts-readme.txt": "https://raw.githubusercontent.com/facebookresearch/fairseq/main/examples/stories/README.md",
    "asap2-paper.pdf": "https://zenodo.org/records/14781349/files/asap_2_paper_preprint.pdf?download=1",
})

def fetch(item):
    name, url = item
    result = {"url": url, "path": "evidence/" + name,
              "retrieved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OpenTextDetectorSourceResearch/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read(5_000_001)
            if len(body) > 5_000_000:
                raise ValueError("Metadata size limit exceeded")
            result.update(http_status=response.status, final_url=response.url,
                          content_type=response.headers.get("Content-Type"))
        (DEST / name).write_bytes(body)
        result.update(status="retrieved", bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        if name.startswith("common-pile"):
            meta = json.loads(body)
            result.update(dataset_id=meta.get("id"), dataset_revision=meta.get("sha"),
                          repository_file_count=len(meta.get("siblings", [])))
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return result

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, URLS.items()))
    (ROOT / "evidence-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    for r in records:
        print(r["path"], r["status"], r.get("bytes", r.get("error")))
