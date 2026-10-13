"""Normalize released test data and explicitly marked local proxies; no inference."""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
RAW = B / "data/raw"
SEED = "pangram4-local-v1"


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def readjsonl(p):
    with p.open() as f:
        for line in f:
            yield json.loads(line)


def record(dataset, id, text, label, **meta):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty text {dataset}/{id}")
    return {
        "id": f"{dataset}:{id}",
        "dataset": dataset,
        "text": text,
        "label": label,
        "language": "en",
        "text_sha256": digest(text),
        "group_id": str(meta.pop("group_id", id)),
        **meta,
    }


def sample(rows, n, keys=("cohort", "label", "generator", "domain", "attack")):
    pools = defaultdict(list)
    counts = Counter()
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in keys)
        counts[key] += 1
        pool = pools[key]
        pool.append((digest(SEED + r["id"]), r))
        if n and len(pool) > n:
            pool.sort(key=lambda x: x[0])
            pool.pop()
    result = [
        r for key in sorted(pools) for _, r in sorted(pools[key], key=lambda x: x[0])
    ]
    return result, {
        "eligible": sum(counts.values()),
        "selected": len(result),
        "strata": len(counts),
        "per_stratum": n,
        "keys": keys,
        "selection": "lowest SHA256(seed + id), independent of detector scores",
    }


def liang():
    for p in sorted((RAW / "liang").glob("*.json")):
        if "receipt" in p.name:
            continue
        for i, r in enumerate(json.loads(p.read_text())):
            polish = "polished" in p.stem
            yield record(
                "liang",
                p.stem + ":" + str(i),
                r["document"],
                "human",
                cohort=p.stem,
                task="polish" if polish else "human",
                group_id=("toefl" if "TOEFL" in p.stem else "hewlett") + str(i),
                provenance="released text; polish retains human base label under minor-edit convention",
            )


def pelic():
    # Historical timestamp and >=50 words are local eligibility rules, not Pangram's unpublished selection.
    for r in csv.DictReader((RAW / "pelic/answer.csv").open()):
        if (
            not r["text"].strip()
            or len(r["text"].split()) < 50
            or r["created_date"][:4] >= "2022"
        ):
            continue
        yield record(
            "pelic",
            r["answer_id"],
            r["text"],
            "human",
            cohort="historical_50plus_words",
            domain="learner_english",
            group_id=r["anon_id"],
            task="human",
            provenance="PELIC original text; pre-2022 timestamp; >=50 words; multiple revisions may share an author",
        )


def meld():
    for r in readjsonl(RAW / "meld.jsonl"):
        yield record(
            "meld_eval",
            r["id"],
            r["text"],
            {0: "human", 1: "ai"}[r["label"]],
            cohort="released",
            generator=r["generator"],
            domain=r["domain"],
            attack=r["attack"],
            group_id=r["prompt_id"],
        )


def detectrl():
    for p in sorted((RAW / "detectrl").rglob("*test.json")):
        for i, r in enumerate(json.loads(p.read_text())):
            if r["label"] not in ["human", "llm"]:
                raise ValueError(f"Unexpected DetectRL label {r['label']}")
            yield record(
                "detectrl",
                str(p.relative_to(RAW)) + ":" + str(i),
                r["text"],
                {"human": "human", "llm": "ai"}[r["label"]],
                cohort=str(p.relative_to(RAW / "detectrl")).removesuffix(".json"),
                domain=r.get("data_type"),
                generator=r.get("llm_type"),
                group_id=digest(r["text"]),
                provenance="native labels retained, including transformed-human conditions",
            )


def epoch():
    with zipfile.ZipFile(RAW / "epoch.zip") as z:
        for path in sorted(z.namelist()):
            p = "/".join(path.split("/")[1:])
            parts = p.split("/")
            if p.startswith("corpus/") and re.search(r"/snippet_\d+\.txt$", p):
                cohort = "human"
                domain = parts[1]
                author = parts[2]
                gen = "human"
                label = "human"
            elif p.startswith(
                ("stages/vanilla/", "stages/style_transfer/")
            ) and p.endswith(".txt"):
                cohort = parts[1]
                domain = parts[2]
                author = parts[3]
                gen = Path(p).stem
                label = "ai"
            else:
                continue
            yield record(
                "epoch",
                p,
                z.read(path).decode(),
                label,
                cohort=cohort,
                domain=domain,
                generator=gen,
                group_id=domain + "/" + author,
            )


def opai():
    for p in sorted((RAW / "opai").glob("*.csv")):
        for r in csv.DictReader(p.open()):
            if r["split"] != "test":
                continue
            fraction = float(r["ai_char_ratio"])
            spans = json.loads(r["ai_spans_char"])
            # Treat intermediate trajectories as mixed regardless of arbitrary binary threshold.
            label = "human" if fraction == 0 else "ai" if fraction == 1 else "mixed"
            yield record(
                "opai",
                r["record_id"],
                r["text"],
                label,
                cohort=r["version"],
                version=r["version"],
                generator=r["generator"],
                domain=r["domain"],
                target_fraction=fraction,
                ai_spans=spans,
                group_id=r["document_hash_id"],
                task="progressive_edit",
                operation=r["edit_operation"],
            )


def sem():
    import pyarrow.parquet as pq

    for r in pq.read_table(RAW / "sem.parquet").to_pylist():
        if r["split"] != "test":
            continue
        yield record(
            "sem_detect",
            ":".join([r["paper_id"], r["author"], r["review_id"]]),
            r["clean_review"],
            {"human": "human", "ai": "ai", "rewrite": "mixed"}[r["class"]],
            cohort=r["class"],
            generator=r["author"],
            domain="peer_review",
            group_id=r["paper_id"],
            task="edited" if r["class"] == "rewrite" else "binary",
            provenance="ICLR 2022 official test subset; refined reviews separately reported, excluded from pure binary metrics",
        )


def gede():
    p = RAW / "gede.db"
    if not p.exists():
        with zipfile.ZipFile(RAW / "gede.zip") as z:
            p.write_bytes(z.read("database.db"))
    with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as c:
        query = """select a.id,a.answer,a.is_human,a.rewrite_of,j.prompt_mode,j.model,d.name,a.question_id
                 from answers a left join jobs j on a.job_id=j.id
                 join questions q on a.question_id=q.id join datasets d on q.dataset_id=d.id"""
        for id, text, human, parent, mode, model, domain, q in c.execute(query):
            if not text or not text.strip():
                continue
            # Source benchmark treats all non-human rows as positive, even improved essays.
            yield record(
                "gede",
                id,
                text,
                "human" if human else "ai",
                cohort="human" if human else mode,
                generator=model or "human",
                domain=domain,
                group_id=str(q),
                task="polish" if mode == "improve-human" else "binary",
                provenance="native binary positive convention; improve-human also reported with polish false-alarm metric",
            )


def saha():
    manifest = RAW / "saha/selected.json"
    if not manifest.exists():
        return
    for item in json.loads(manifest.read_text()):
        yield record(
            "saha",
            item["path"],
            (RAW / item["file"]).read_text(),
            item["label"],
            cohort=item["cohort"],
            generator=item["generator"],
            domain="peer_review",
            task=item["task"],
            group_id=item["path"],
        )


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def local(n):
    books = list(readjsonl(ROOT / "research/data/human_pg19/books.jsonl"))
    ai = list(readjsonl(ROOT / "research/data/ai_mdta_2025/responses.jsonl"))
    # One centered excerpt per book avoids treating highly correlated passages as independent books.
    humans = []
    for b in books:
        matches = list(re.finditer(r"\S+", b["text"]))
        mid = len(matches) // 2
        end = min(len(matches), mid + 500)
        t = b["text"][matches[mid].start() : matches[end - 1].end()]
        humans.append(
            record(
                "local_binary",
                b["id"],
                t,
                "human",
                cohort="pg19_center_500_words",
                domain="books",
                group_id=b["id"],
            )
        )
    machines = [
        record(
            "local_binary",
            r["id"],
            r["text"],
            "ai",
            cohort="mdta_2025",
            domain=r["domain"],
            generator=r["generator"],
            group_id=r["domain"] + ":" + str(r["question_index"]),
        )
        for r in ai
    ]
    base, _ = sample(humans + machines, n)
    out = list(base)
    for r in base:
        for words in [40, 100, 250]:
            matches = list(re.finditer(r"\S+", r["text"]))
            if len(matches) < words:
                continue
            out.append(
                record(
                    "local_length",
                    r["id"] + f":{words}",
                    r["text"][: matches[words - 1].end()],
                    r["label"],
                    cohort=str(words),
                    domain=r["domain"],
                    generator=r.get("generator"),
                    group_id=r["group_id"],
                )
            )
    # Mix complete sentences; select AI sources long enough for every requested block, without padding/repetition.
    long_ai = [r for r in machines if len(sentences(r["text"])) >= 20]
    long_ai.sort(key=lambda r: digest(SEED + r["id"]))
    for i, h in enumerate(
        sorted(humans, key=lambda r: digest(SEED + r["id"]))[: n or 100]
    ):
        a = long_ai[i % len(long_ai)]
        hs = sentences(h["text"])
        ais = sentences(a["text"])
        # Use the full source book for long block sizes.
        book = next(b for b in books if b["id"] == h["group_id"])
        hs = sentences(book["text"])
        hs = hs[len(hs) // 2 :]
        for block in [1, 4, 8, 12, 16, 20]:
            chunks = [("human", " ".join(hs[:block])), ("ai", " ".join(ais[:block]))]
            if i % 2:
                chunks.reverse()
            text = ""
            spans = []
            for label, t in chunks:
                if text:
                    text += "\n\n"
                start = len(text)
                text += t
                if label == "ai":
                    spans.append([start, len(text)])
            out.append(
                record(
                    "local_mixed",
                    f"{i}:{block}",
                    text,
                    "mixed",
                    cohort=str(block),
                    block_size=block,
                    group_id=h["group_id"] + "+" + a["group_id"],
                    ai_spans=spans,
                    target_fraction=sum(e - s for s, e in spans) / len(text),
                    provenance="synthetic two-block proxy; unmatched topics and regex sentence splitting; not Pangram reconstruction",
                )
            )
    return out


def documents(dataset):
    import xml.etree.ElementTree as ET

    manifest = RAW / "document-manifest.json"
    if not manifest.exists():
        return
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for item in json.loads(manifest.read_text()):
        if item["dataset"] != dataset:
            continue
        path = RAW / item["file"]
        if (
            item.get("source_sha256")
            and hashlib.sha256(path.read_bytes()).hexdigest() != item["source_sha256"]
        ):
            raise ValueError("Publisher checksum mismatch: " + str(path))
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        paragraphs = [
            "".join(t.text or "" for t in p.findall(".//w:t", ns))
            for p in root.findall(".//w:body//w:p", ns)
        ]
        text = "\n\n".join(p for p in paragraphs if p.strip())
        if not text.strip():
            raise ValueError("Empty document " + item["file"])
        yield record(
            dataset,
            item["id"],
            text,
            item["label"],
            cohort=item["cohort"],
            domain="academic" if dataset == "vub" else "short_writing",
            provenance="Word body paragraphs and tables, in XML order; headers/footers and formatting excluded",
        )


ADAPTERS = {
    "pelic": pelic,
    "vub": lambda: documents("vub"),
    "perkins": lambda: documents("perkins"),
    "liang": liang,
    "meld_eval": meld,
    "detectrl": detectrl,
    "epoch": epoch,
    "opai": opai,
    "sem_detect": sem,
    "gede": gede,
    "saha": saha,
}


def validate(rows):
    ids = set()
    for r in rows:
        if r["id"] in ids:
            raise ValueError("Duplicate ID " + r["id"])
        ids.add(r["id"])
        if r["label"] not in ("human", "ai", "mixed"):
            raise ValueError("Invalid label")
        last = 0
        for s, e in r.get("ai_spans", []):
            if not 0 <= s < e <= len(r["text"]) or s < last:
                raise ValueError("Invalid spans " + r["id"])
            last = e
        count = len(r["text"].split())
        r["words"] = count
        r["length_bucket"] = (
            "<50"
            if count < 50
            else "50-99"
            if count < 100
            else "100-249"
            if count < 250
            else "250-499"
            if count < 500
            else "500+"
        )
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-stratum", type=int, default=4)
    p.add_argument("--only", default="all", choices=["all", "local", *ADAPTERS])
    a = p.parse_args()
    if a.per_stratum < 0:
        p.error("per-stratum must be nonnegative (0 means all)")
    out = B / "data/prepared"
    out.mkdir(parents=True, exist_ok=True)
    summary = (
        json.loads((out / "manifest.json").read_text()).get("datasets", {})
        if (out / "manifest.json").exists()
        else {}
    )
    for name, fn in ADAPTERS.items():
        if a.only not in ("all", name):
            continue
        rows, info = sample(fn(), a.per_stratum)
        validate(rows)
        path = out / (name + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        info["unique_texts"] = len({r["text_sha256"] for r in rows})
        info["cohorts"] = dict(Counter(r.get("cohort", "unknown") for r in rows))
        summary[name] = info
        print(name, info, flush=True)
    if a.only in ("all", "local"):
        rows = validate(local(a.per_stratum))
        path = out / "local.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        summary["local"] = {
            "selected": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out / "manifest.json").write_text(
        json.dumps({"seed": SEED, "datasets": summary}, indent=2) + "\n"
    )
