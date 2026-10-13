"""EditLens preprocessing with source offsets retained for passage highlighting.

Adapted from Pangram Labs' EditLens scripts/preprocess.py (CC BY-NC-SA 4.0).
See docs/THIRD_PARTY.md. Removed preambles remain unhighlighted.
"""
import re
import emoji

STARTS = ("Sure", "Here", "Abstract", "Title", "I'm happy to help", "Certainly")


def preprocess(text):
    chars, mapping, position = [], [], 0
    for item in emoji.emoji_list(text):
        start, end = item["match_start"], item["match_end"]
        for i in range(position, start):
            chars.append(text[i])
            mapping.append((i, i+1))
        replacement = emoji.demojize(item["emoji"])
        chars.extend(replacement)
        mapping.extend([(start, end)] * len(replacement))
        position = end
    for i in range(position, len(text)):
        chars.append(text[i])
        mapping.append((i, i+1))
    value = "".join(chars)
    if "</think>" in value:
        first = value.index("</think>") + len("</think>")
        last = value.find("</think>", first)
        last = len(value) if last == -1 else last
        value, mapping = value[first:last], mapping[first:last]
        left, right = len(value) - len(value.lstrip()), len(value.rstrip())
        value, mapping = value[left:right], mapping[left:right]
    lines = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", value) if m.group().strip()]
    if lines:
        first_line = re.sub(r"^[^a-zA-Z0-9]*", "", value[lines[0][0]:lines[0][1]])
        first_line = emoji.replace_emoji(first_line, "")
        if first_line.startswith(STARTS) and len(lines) > 1:
            pieces, maps = [], []
            for start, end in lines[1:]:
                if pieces:
                    pieces.append("\n")
                    maps.append(mapping[start])
                pieces.append(value[start:end])
                maps.extend(mapping[start:end])
            value, mapping = "".join(pieces), maps
    chars, maps = [], []
    for char, span in zip(value, mapping):
        lower = char.lower()
        chars.extend(lower)
        maps.extend([span] * len(lower))
    value, mapping = "".join(chars), maps
    chars, maps = [], []
    for match in re.finditer(r"\s+|\S+", value):
        start, end = match.span()
        if match.group().isspace():
            if chars:
                chars.append(" ")
                maps.append((mapping[start][0], mapping[end-1][1]))
        else:
            chars.extend(match.group())
            maps.extend(mapping[start:end])
    if chars and chars[-1] == " ":
        chars.pop()
        maps.pop()
    return "".join(chars), maps
