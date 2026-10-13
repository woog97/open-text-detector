"""Count whitespace-separated words in a saved JSONL document export."""
import argparse
import json
from pathlib import Path


def count_words(path):
    documents = 0
    words = 0
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            document = json.loads(line)
            words += len(document["text"].split())
            documents += 1
    return {
        "documents": documents,
        "total_words": words,
        "average_words_per_document": words / documents if documents else 0,
        "method": "Whitespace-separated words: len(text.split())",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    print(json.dumps(count_words(args.export), indent=2))
