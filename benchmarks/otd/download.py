"""Download pinned public data only. No inference APIs or upstream code execution."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import httpx

B = Path(__file__).resolve().parent
RAW = B / "data" / "raw"


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(name, url, cap=1_000_000_000):
    p = (RAW / name).resolve()
    if not p.is_relative_to(RAW.resolve()):
        raise ValueError("Download path escapes data directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = p.with_name(p.name + ".receipt.json")
    if p.exists() and meta.exists():
        m = json.loads(meta.read_text())
        if m["url"] == url and m["sha256"] == sha(p):
            return p
        raise ValueError(f"Cache mismatch: {p}")
    tmp = p.with_name(p.name + ".partial")
    size = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as out:
            for chunk in r.iter_bytes():
                size += len(chunk)
                if size > cap:
                    raise ValueError(f"Download exceeds cap: {url}")
                out.write(chunk)
    tmp.replace(p)
    meta.write_text(
        json.dumps({"url": url, "bytes": size, "sha256": sha(p)}, indent=2) + "\n"
    )
    print("Downloaded", name, size, flush=True)
    return p


def jobs():
    result = []

    def gh(name, repo, rev, path):
        result.append((name, f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"))

    def hf(name, repo, rev, path):
        result.append(
            (name, f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}")
        )

    result.append(
        (
            "pelic/answer.csv",
            "https://media.githubusercontent.com/media/ELI-Data-Mining-Group/PELIC-dataset/c4526baeb8fb5d69732f9e2a8e1430b41ed38c53/corpus_files/answer.csv",
        )
    )
    for name in [
        "TOEFL_real_91",
        "TOEFL_gpt4polished_91",
        "HewlettStudentEssay_real_88",
    ]:
        gh(
            "liang/" + name + ".json",
            "Weixin-Liang/ChatGPT-Detector-Bias",
            "1d05ed8242d4694a0ed7b73a79c52be8864470f9",
            "Data_and_Results/Human_Data/" + name + "/data.json",
        )
    hf(
        "meld.jsonl",
        "anon-review-meld-2026/meld-eval",
        "4c80f0fdc003854e5b6bfda90f7ac41ffb12e232",
        "meld_eval.jsonl",
    )
    for domain in ["essays", "news", "reports", "abstracts"]:
        hf(
            "opai/" + domain + "_qwen.csv",
            "OpAI-Bench1/OpAI-Bench",
            "4fb59011bcab8c62863b7083fe0f0a3345555558",
            f"default/test/{domain}_qwen3-8b.csv",
        )
    for gen in ["gpt-5.4", "gemini-2.5-flash"]:
        hf(
            "opai/abstracts_" + gen + ".csv",
            "OpAI-Bench1/OpAI-Bench",
            "4fb59011bcab8c62863b7083fe0f0a3345555558",
            f"default/test/abstracts_{gen}.csv",
        )
    hf(
        "sem.parquet",
        "Sem-Detect/ML_Conferences-Peer-Reviews",
        "18b74b551670bff6b04e57570232a1694997863b",
        "reviews/iclr_2022-00000-of-00001.parquet",
    )
    gh(
        "gede.zip",
        "lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts",
        "fc84dfe33619eefbc3a6014c191ad9a43f8543cc",
        "database/database.db.zip",
    )
    result.append(
        (
            "epoch.zip",
            "https://codeload.github.com/jaeholee-brown/ai-text-detectors/zip/3f5ed200e212ef5ca957b6f5675d99ed29ff3c3c",
        )
    )
    d = json.loads((B / "sources/detectrl-metadata.json").read_text())
    for x in d["tree"]:
        path = x["path"]
        if (
            path.startswith("Benchmark/Tasks/")
            and path.endswith("test.json")
            and "/Task2/" not in path
        ):
            gh(
                "detectrl/" + path.removeprefix("Benchmark/Tasks/"),
                "NLP2CT/DetectRL",
                d["sha"],
                path,
            )
    for name in ["easy", "hard", "humans", "humanized"]:
        gh(
            "saha/paths_" + name + ".txt",
            "FLAIR-IISc/ai-in-peer-review",
            "cffc6649c4e73d4699791823155f2df8b80c2eb6",
            f"PathFiles/all_paths_{name}.txt",
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="all")
    a = p.parse_args()
    selected = [j for j in jobs() if a.only == "all" or j[0].startswith(a.only)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        fs = [ex.submit(fetch, *j) for j in selected]
        for f in fs:
            f.result()
