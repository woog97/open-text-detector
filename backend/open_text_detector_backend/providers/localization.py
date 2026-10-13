"""Offset-preserving evidence localization; no inferred authorship probabilities."""
import math
import re


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-80, min(80, value))))


def window_starts(length, capacity, overlap):
    if length <= capacity:
        return [0]
    starts = list(range(0, length - capacity + 1, capacity - overlap))
    if starts[-1] != length - capacity:
        starts.append(length - capacity)
    return starts


def localize(text, offsets, scores, threshold):
    # Byte-level tokenizers can assign several tokens to one Unicode character.
    # Coalesce overlapping offsets so every displayed character appears once.
    tokens = []
    for (start, end), value in zip(offsets, scores, strict=True):
        if end <= start:
            continue
        if tokens and start < tokens[-1]["end"]:
            item = tokens[-1]
            item["raw_score"] += value
            item["token_count"] += 1
            item["end"] = max(item["end"], end)
        else:
            tokens.append({"start": start, "end": end, "raw_score": value, "token_count": 1})
    for item in tokens:
        item["raw_score"] /= item["token_count"]
        item["score"] = sigmoid(item["raw_score"])
        item["label"] = "ai_evidence" if item["raw_score"] > threshold else "low_evidence"
    # Sentence boundaries are presentation units, not separately scored inputs.
    boundaries = [0] + [m.end() for m in re.finditer(r'[.!?][\"\u201d\u2019\')\]]*(?:\s+|$)|\n\s*\n', text)]
    boundaries = sorted(set(boundaries + [len(text)]))
    if len(boundaries) > 5001:
        step = math.ceil((len(boundaries) - 1) / 5000)
        boundaries = boundaries[::step] + ([len(text)] if boundaries[::step][-1] != len(text) else [])
    segments, idx = [], 0
    for start, end in zip(boundaries, boundaries[1:]):
        values = []
        while idx < len(tokens) and (tokens[idx]["start"] + tokens[idx]["end"]) / 2 < end:
            values.append(tokens[idx])
            idx += 1
        count = sum(t["token_count"] for t in values)
        raw = sum(t["raw_score"] * t["token_count"] for t in values) / count if count else 0.0
        segments.append({"start": start, "end": end, "raw_score": raw, "score": sigmoid(raw),
                         "token_count": count,
                         "label": "ai_evidence" if count and raw > threshold else "low_evidence"})
    return tokens, segments
