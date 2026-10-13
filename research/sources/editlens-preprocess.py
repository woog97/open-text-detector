import re
import emoji


BOILERPLATE_STARTS = [
    "Sure",
    "Here",
    "Abstract",
    "Title",
    "I'm happy to help",
    "Certainly",
]


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def normalize_emoji(text):
    return emoji.demojize(text)


def remove_think_tag(text):
    if "</think>" in text:
        text = text.split("</think>")[1].strip()
    return text


def remove_ai_header(text):
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) == 0:
        return text
    first_paragraph = paragraphs[0]
    first_paragraph = re.sub(r"^[^a-zA-Z0-9]*", "", first_paragraph)
    first_paragraph = emoji.replace_emoji(first_paragraph, "")
    if any(first_paragraph.startswith(phrase) for phrase in BOILERPLATE_STARTS):
        if len(paragraphs) > 1:
            text = "\n".join(paragraphs[1:])
    return text


def clean_text(text):
    text = normalize_emoji(text)
    text = remove_think_tag(text)
    text = remove_ai_header(text)
    text = text.lower()
    text = normalize_whitespace(text)
    return text


def count_words(text):
    return len(re.findall(r"\b\w+\b", text))


def score_to_bucket(score, n_buckets, lo_threshold, hi_threshold):
    """
    Assign a score in [0, 1] to one of n_buckets.

    Bucket 0: scores <= lo_threshold
    Bucket n-1: scores >= hi_threshold
    Buckets 1 to n-2: evenly spaced between (lo_threshold, hi_threshold)
    """
    if n_buckets == 2:
        midpoint = (lo_threshold + hi_threshold) / 2
        return 0 if score <= midpoint else 1
    if score <= lo_threshold:
        return 0
    elif score >= hi_threshold:
        return n_buckets - 1
    else:
        normalized = (score - lo_threshold) / (hi_threshold - lo_threshold)
        return 1 + int(normalized * (n_buckets - 2))
