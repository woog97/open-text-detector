"""Pinned stratified peer-review subset from publisher path lists."""

import concurrent.futures
import hashlib
import json
import re
from urllib.parse import quote

from download import RAW, fetch

REV = "cffc6649c4e73d4699791823155f2df8b80c2eb6"


def main():
    from collections import defaultdict

    groups = defaultdict(list)
    for subset in ["easy", "hard", "humans"]:
        for path in (RAW / f"saha/paths_{subset}.txt").read_text().splitlines():
            path = path.strip()
            if not path.endswith(".txt") or "_keypoints" in path:
                continue
            m = re.search(r"/level([1-4])/", path)
            level = m.group(1) if m else "human"
            if subset != "humans" and not m:
                continue
            cohort = subset + "/" + level
            groups[cohort].append(path)
    selected = []
    for cohort, paths in sorted(groups.items()):
        for path in sorted(
            set(paths),
            key=lambda x: hashlib.sha256(
                ("pangram4-local-v1" + x).encode()
            ).hexdigest(),
        )[:8]:
            level = cohort.split("/")[-1]
            parts = path.split("/")
            i = next(
                (
                    i
                    for i, p in enumerate(parts)
                    if p.startswith("level") and len(p) == 6
                ),
                None,
            )
            generator = (
                "human"
                if i is None
                else parts[i + 1]
                if "/hard-subset/" in path
                else parts[i - 1]
            )
            filename = (
                "saha/texts/" + hashlib.sha256(path.encode()).hexdigest() + ".txt"
            )
            selected.append(
                {
                    "path": path,
                    "file": filename,
                    "cohort": cohort,
                    "generator": generator,
                    "label": "human" if level in ["human", "4"] else "ai",
                    "task": "polish" if level == "4" else "binary",
                }
            )

    def get(r):
        fetch(
            r["file"],
            "https://raw.githubusercontent.com/FLAIR-IISc/ai-in-peer-review/"
            + REV
            + "/data/"
            + quote(r["path"], safe="/"),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for f in ex.map(get, selected):
            pass
    (RAW / "saha/selected.json").write_text(json.dumps(selected, indent=2) + "\n")


if __name__ == "__main__":
    main()
