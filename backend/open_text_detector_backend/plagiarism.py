"""Exact phrase overlap against the user's reference corpus, not a web plagiarism index."""
import re


def words(text):
    return [(m.group().casefold(), m.start(), m.end()) for m in re.finditer(r"\w+", text)]


def compare_corpus(text, sources, n=8):
    tokens = words(text)
    index = {}
    for i in range(max(0, len(tokens) - n + 1)):
        phrase = tuple(t[0] for t in tokens[i:i+n])
        index.setdefault(phrase, []).append(i)
    covered = set()
    matches = []
    for source in sources:
        source_tokens = words(source["text"])
        positions = set()
        for i in range(max(0, len(source_tokens) - n + 1)):
            for position in index.get(tuple(t[0] for t in source_tokens[i:i+n]), []):
                positions.update(range(position, position+n))
        covered.update(positions)
        spans = []
        for i in sorted(positions):
            start, end = tokens[i][1:]
            if spans and start <= spans[-1]["end"] + 3:
                spans[-1]["end"] = end
            else:
                spans.append({"start": start, "end": end})
        if positions:
            matches.append({"source_id": source["id"], "title": source["title"],
                            "source_url": source.get("source_url"), "matched_words": len(positions),
                            "spans": spans})
    return {"scope": "workspace_corpus_only", "sources_checked": len(sources),
            "matched_word_fraction": len(covered) / max(1, len(tokens)),
            "matches": sorted(matches, key=lambda m: -m["matched_words"]),
            "notice": "Exact phrase overlap with your reference documents; not an internet-wide plagiarism check."}
