"""Find bibliographies in extracted paper text, so paper-level flagged shares can leave reference entries out.

A bibliography starts at a "References"/"Bibliography" heading line whose next lines look like reference entries,
and runs while lines keep carrying citation signals (years, arXiv, URLs, DOIs, page ranges, venue words). It ends after
the last signal line before the first appendix-opening line (APPENDIX) or before a stretch of at least GAP_LINES lines
and GAP_CHARS characters without any signal (unusual appendices, or the paper's last pages).
"""
import re

START = re.compile(r'^[ \t]*(?:\d{1,2}\.?[ \t]+)?(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|Literature Cited|LITERATURE CITED|Works Cited)[ \t]*:?[ \t]*$', re.M)
SIGNAL = re.compile(r'\b(?:19|20)\d\d[a-z]?\b|arXiv|https?://|\bURL\b|\bdoi\b|\bpp\.|Proceedings|Conference|Journal|Transactions|Advances in|\bIn [A-Z]|et al\.|preprint|Press|\bvol\.|\bVol\.')
GAP_LINES, GAP_CHARS = 5, 400
# Lines that open an appendix in extracted text: a lone section letter ("A", "A.1"), "Appendix ...", or a capitalised
# lettered heading ("A EXPERIMENTAL DETAILS", "B.2 Proof of Lemma 3"). Reference entries do not take these shapes.
APPENDIX = re.compile(r'^[ \t]*(?:[A-H](?:\.\d{1,2})*\.?|(?:APPENDI(?:X|CES)|Appendi(?:x|ces)|SUPPLEMENTARY MATERIALS?|Supplementary Materials?)\b[^\n]{0,80}|[A-H](?:\.\d{1,2})*\.?[ \t]+[A-Z][A-Z0-9 \-:,&()/]{3,90})[ \t]*$')
AUTHORS = re.compile(r'^(?:[^,]{2,40},){3,}')  # long author lists: many short comma-separated names


def _looks_like_refs(chunk):
    lines = [l for l in chunk.split('\n')[:20] if l.strip()]
    return len(lines) >= 3 and sum(bool(SIGNAL.search(l)) for l in lines) >= max(3, len(lines) // 3)


def _end(text, a):
    """End of the last citation-signal line before a long signal-free stretch (or the end of the text)."""
    last = a; gap_lines = gap_chars = 0; pos = a
    for line in text[a:].split('\n'):
        end = pos + len(line)
        if APPENDIX.match(line):
            return max(last, pos - 1) if gap_lines == 0 or gap_chars < GAP_CHARS else last  # up to the appendix line
        if SIGNAL.search(line) or AUTHORS.match(line):
            last = end; gap_lines = gap_chars = 0
        elif line.strip():
            gap_lines += 1; gap_chars += len(line)
            if gap_lines >= GAP_LINES and gap_chars >= GAP_CHARS:
                break
        pos = end + 1
    return last


def bibliography_spans(text):
    """[(start, end)] character spans (heading included) of every bibliography in the text."""
    spans = []
    for m in START.finditer(text):
        if spans and m.start() < spans[-1][1]:
            continue
        a = m.end()
        if _looks_like_refs(text[a:a + 4000]):
            spans.append((m.start(), _end(text, a)))
    return spans


def in_spans(start, spans):
    return any(a <= start < b for a, b in spans)


# Counting modes for paper-level flagged shares: name -> (include bibliography, include appendices).
MODES = {'all': (True, True), 'nobib': (False, True), 'noapp': (True, False), 'body': (False, False)}


def sections(text):
    """(bibliography spans, appendix span or None). The appendix is everything after the first bibliography
    (appendices, checklists, statements); later bibliographies inside it stay bibliography. Papers that place their
    appendix before the references have it counted as body."""
    spans = bibliography_spans(text)
    tail = (spans[0][1], len(text)) if spans and len(text) - spans[0][1] > 200 else None
    return spans, tail


def section_of(start, spans, tail):
    """'bib', 'app' or 'body' for a sentence starting at `start`."""
    if in_spans(start, spans): return 'bib'
    if tail and tail[0] <= start < tail[1]: return 'app'
    return 'body'


def included(section, mode):
    inc_bib, inc_app = MODES[mode]
    return section == 'body' or (section == 'bib' and inc_bib) or (section == 'app' and inc_app)
