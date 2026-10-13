"""Retrieve released VUB and Perkins document stimuli, never detector reports."""

import concurrent.futures
import json

import httpx

from download import RAW, B, fetch


def main():
    url = json.loads((B / "sources/vub-results.json").read_text())["data"][0][
        "relationships"
    ]["files"]["links"]["related"]["href"]
    r = httpx.get(url + "&page[size]=100&sort=name", timeout=60)
    r.raise_for_status()
    d = r.json()
    files = list(d["data"])
    while d["links"].get("next"):
        r = httpx.get(d["links"]["next"], timeout=60)
        r.raise_for_status()
        d = r.json()
        files += d["data"]
    files = list({f["id"]: f for f in files}.values())
    (B / "sources/vub-all-files.json").write_text(json.dumps(files, indent=2))
    jobs = []
    items = []
    for f in files:
        name = f["attributes"]["name"]
        if name.startswith("AI") and name.lower().endswith(".docx"):
            file = "vub/" + name
            jobs.append((file, f["links"]["download"]))
            items.append(
                {
                    "dataset": "vub",
                    "file": file,
                    "id": f["id"],
                    "label": "ai",
                    "cohort": "fully_ai",
                }
            )
    folders = json.loads((B / "sources/perkins-folders.json").read_text())
    for folder in folders:
        if not folder.get("parent_id"):
            continue
        u = f"https://data.mendeley.com/public-api/datasets/xv6fk2mmh9/files?folder_id={folder['id']}&version=3"
        r = httpx.get(
            u,
            headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
            timeout=60,
        )
        r.raise_for_status()
        for f in r.json():
            if not f["filename"].lower().endswith(".docx"):
                continue
            file = "perkins/" + folder["id"] + "/" + f["filename"]
            jobs.append((file, f["content_details"]["download_url"]))
            items.append(
                {
                    "dataset": "perkins",
                    "file": file,
                    "id": f["id"],
                    "label": "human" if "Control" in folder["name"] else "ai",
                    "cohort": folder["name"],
                    "source_sha256": f["content_details"]["sha256_hash"],
                }
            )

    def get(j):
        return fetch(*j, cap=10_000_000)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(get, jobs))
    (RAW / "document-manifest.json").write_text(json.dumps(items, indent=2) + "\n")
    print("Document files", len(items))


if __name__ == "__main__":
    main()
