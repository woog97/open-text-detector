"""Label baseline words as figure text or math, and render text with omission markers.

Prototype (October 2026). Signals come from the PDF itself, not a model:
- fonts: each baseline word takes the font of the PDF characters under it; math
  fonts (CMMI, CMSY, TeX-matha, ...) mark math even when their text is garbled;
- drawings and images: dense clusters of vector paths or embedded images are
  figures; words inside them (and figure-style fonts next to them) are figure text.
Captions ("Figure 3: ...") are kept. Omitted text becomes a marker so a labeler
or the training loss can recognise it, and every omission keeps its source words.
"""
from collections import Counter, defaultdict
import re

MARKERS = {'figure': '⟦figure omitted⟧', 'math_display': '⟦equation omitted⟧',
           'math_inline': '⟦math⟧', 'table': '⟦table omitted⟧', 'numeric': '⟦numbers omitted⟧',
           'hidden': '', 'duplicate': '', 'references': '⟦references omitted⟧'}   # invisible or doubled text is dropped without a marker
DEFAULT_POLICY = {'figure': True, 'math_display': True, 'math_inline': True, 'table': True, 'numeric': True,
                  'hidden': True, 'duplicate': True, 'references': True}
# A tick label or table cell: a signed/decimal/scientific number, optionally with a
# percent sign, multiplier or magnitude suffix (k, M, B), or a bracketed/± pair piece.
NUMERIC = re.compile(r"^[(\[]?[−\-–+±~≈<>]?\$?\d[\d.,]*(e[−\-+]?\d+)?[%×x]?([kKMBGT]|ms|s)?[)\],;:]?$|^[−\-–+±]?\.\d+$|^[−\-–]$")

MATH_FONT = re.compile(
    r'^(CM(MI|SY|EX|BSY|MIB)|MSAM|MSBM|EU(FM|SM|EX)|RSFS|STIX(Math|TwoMath|General|NonUnicode|Size|Variants)'
    r'|LMMath|LatinModernMath|Cambria-?Math|TeX-?math|.*TeXGyre.*Math|t?xmi|t?xsy|t?xex|txsys|p?xmi|pxsy|ntx(mi|sy|ex)'
    r'|NewTX(MI|SY)|NewPX(MI|SY)|NewCMMath|XITSMath|Asana|DejaVuMath|Symbol|MT(MI|SY|EX)|rtxmi|zxmi|esint|wasy|stmary'
    r'|bbm|dsrom|MnSymbol|Fourier.*Math|MathJax)', re.I)
MONO_FONT = re.compile(r'Mon|Courier|Consol|Inconsolata|Menlo|SourceCodePro|FiraMono|LMMono|CMTT|Typewriter', re.I)
CM_TEXT = re.compile(r'^(CMR|CMBX|CMTI|CMSL|CMSS|LMRoman|CMU)', re.I)
CAPTION = re.compile(r'^\s*(Figure|Fig\.|Table)\s*[A-Z]?\d+', re.I)


def base_font(name):
    return re.sub(r'^[A-Z]{6}\+', '', name or '')


def family(name):
    return re.split(r'[-,]', base_font(name))[0]


def page_features(doc):
    """Per page: characters with fonts, figure-candidate regions, table-like regions."""
    pages = []
    for pg in doc:
        w, h = pg.rect.width, pg.rect.height
        chars = []
        for b in pg.get_text('rawdict')['blocks']:
            for line in b.get('lines', []):
                for s in line['spans']:
                    f = base_font(s['font'])
                    for c in s['chars']:
                        if c['c'].strip():
                            x0, y0, x1, y1 = c['bbox']
                            chars.append(((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, f))
        shapes = []; rules = []
        for d in pg.get_drawings():
            x0, y0, x1, y1 = d['rect']
            if (x1 - x0) > .9 * w or (y1 - y0) > .9 * h: continue          # page frames
            if (y1 - y0) < 1.5 and (x1 - x0) > .2 * w:                      # horizontal rules (booktabs, header)
                rules.append((x0 / w, y0 / h, x1 / w)); continue
            curved = any(it[0] in ('c', 'qu') for it in d['items'])
            filled = d.get('fill') not in (None, (1, 1, 1), [1, 1, 1])
            shapes.append([x0 / w, y0 / h, x1 / w, y1 / h, curved, filled])
        for info in pg.get_image_info():
            x0, y0, x1, y1 = info['bbox']
            if (x1 - x0) * (y1 - y0) > .002 * w * h:
                shapes.append([x0 / w, y0 / h, x1 / w, y1 / h, True, True, 'image'])
        hidden = []
        try:
            for sp in pg.get_texttrace():
                if sp.get('type') == 3 or sp.get('opacity', 1) == 0:     # render mode 3 (invisible) or transparent
                    x0, y0, x1, y1 = sp['bbox']; hidden.append((x0 / w, y0 / h, x1 / w, y1 / h))
        except Exception:
            pass
        tables = _rule_tables(rules)
        images = [s_[:4] for s_ in shapes if len(s_) > 6]
        for t in tables:   # a ruled frame around a raster image is a figure, not a table
            if any(_overlap(t['box'], im) > .3 or _overlap(im, t['box']) > .3 for im in images): t['kind'] = 'figure'
        regions = _cluster(shapes)
        for r in regions:  # shaded rows over a ruled table; whether the text is body font is checked per word later
            if r['kind'] == 'figure' and not r.get('curved') and any(_overlap(r['box'], t['box']) > .5 for t in tables): r['kind'] = 'table'
        pages.append({'chars': chars, 'regions': tables + regions, 'rotation': pg.rotation, 'hidden': hidden})
    return pages


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def _rule_tables(rules):
    """Booktabs-style tables: 2+ horizontal rules with matching extent, stacked closely."""
    rules = sorted(r for r in rules if .06 < r[1] < .95)   # skip running-header rule and page edges
    tables = []; group = []
    def close():
        if len(group) >= 2:
            x0 = min(g[0] for g in group); x1 = max(g[2] for g in group)
            tables.append({'box': [x0, group[0][1] - .004, x1, group[-1][1] + .004], 'kind': 'table', 'paths': len(group)})
    for r in rules:
        if group and (r[1] - group[-1][1] > .35 or abs(r[0] - group[0][0]) > .03 or abs(r[2] - group[0][2]) > .03):
            close(); group = []
        group.append(r)
    close()
    return tables


def _cluster(shapes, gap=.006):
    """Merge nearby shapes; keep clusters that look like figures (or tables)."""
    clusters = []
    for s in sorted(shapes, key=lambda s: (s[1], s[0])):
        box = list(s[:4]); merged = None
        for c in clusters:
            b = c['box']
            if box[0] <= b[2] + gap and box[2] >= b[0] - gap and box[1] <= b[3] + gap and box[3] >= b[1] - gap:
                if merged is None:
                    c['box'] = [min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])]
                    c['n'] += 1; c['curved'] |= s[4]; c['filled'] |= s[5]; c['image'] |= len(s) > 6; merged = c
                else:  # bridge two clusters
                    merged['box'] = [min(merged['box'][0], b[0]), min(merged['box'][1], b[1]), max(merged['box'][2], b[2]), max(merged['box'][3], b[3])]
                    merged['n'] += c['n']; merged['curved'] |= c['curved']; merged['filled'] |= c['filled']; merged['image'] |= c['image']; c['n'] = 0
        clusters = [c for c in clusters if c['n']]
        if merged is None:
            clusters.append({'box': box, 'n': 1, 'curved': s[4], 'filled': s[5], 'image': len(s) > 6})
    out = []
    for c in clusters:
        x0, y0, x1, y1 = c['box']; area = (x1 - x0) * (y1 - y0)
        if area < .008: continue
        if c['image'] or (c['n'] >= 8 and (c['curved'] or c['filled'])) or c['n'] >= 60:
            out.append({'box': c['box'], 'kind': 'figure', 'paths': c['n'], 'curved': c['curved'] or c['image']})
        elif c['n'] >= 4:
            out.append({'box': c['box'], 'kind': 'table', 'paths': c['n']})
    return out


def label_words(artifact, features):
    """Label each baseline word: font class and region. Returns {word_index: dict}."""
    pages = {p['page']: p for p in artifact['pages']}
    text = artifact['text']
    # body font family = most common non-math, non-mono family in the document
    fam = Counter(family(f) for p in features for (_, _, f) in p['chars'] if not MATH_FONT.match(f) and not MONO_FONT.search(f))
    body = fam.most_common(1)[0][0] if fam else ''
    body_is_cm = bool(CM_TEXT.match(body))
    grid = []
    for p in features:
        g = defaultdict(list)
        for cx, cy, f in p['chars']: g[(int(cx * 100), int(cy * 100))].append((cx, cy, f))
        grid.append(g)
    labels = {}; seen = set()
    for i, r in enumerate(artifact['rectangles']):
        k = r['page'] - 1
        if k >= len(features) or features[k]['rotation']: continue
        pw, ph = pages[r['page']]['width'], pages[r['page']]['height']
        x0, y0, x1, y1 = r['x0'] / pw, r['y0'] / ph, r['x1'] / pw, r['y1'] / ph
        fonts = Counter()
        for gx in range(int(x0 * 100) - 1, int(x1 * 100) + 2):
            for gy in range(int(y0 * 100) - 1, int(y1 * 100) + 2):
                for cx, cy, f in grid[k].get((gx, gy), ()):
                    if x0 - .002 <= cx <= x1 + .002 and y0 - .002 <= cy <= y1 + .002: fonts[f] += 1
        if not fonts: continue
        n = sum(fonts.values())
        math_n = sum(v for f, v in fonts.items() if MATH_FONT.match(f) or (not body_is_cm and CM_TEXT.match(f)))
        mono_n = sum(v for f, v in fonts.items() if MONO_FONT.search(f))
        body_n = sum(v for f, v in fonts.items() if family(f) == body)
        cls = 'math' if math_n >= .5 * n else 'mono' if mono_n >= .5 * n else 'body' if body_n >= .5 * n else 'foreign'
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        region = None; region_id = None
        for ri, reg in sorted(enumerate(features[k]['regions']), key=lambda t: t[1]['kind'] != 'table'):
            bx0, by0, bx1, by1 = reg['box']; m = .008 if reg['kind'] == 'figure' else .002
            if bx0 - m <= cx <= bx1 + m and by0 - m <= cy <= by1 + m: region = reg['kind']; region_id = (k, ri); break
        if region is None and cls == 'foreign':
            for reg in features[k]['regions']:
                if reg['kind'] != 'figure': continue
                bx0, by0, bx1, by1 = reg['box']
                if bx0 - .04 <= cx <= bx1 + .04 and by0 - .04 <= cy <= by1 + .04: region = 'figure_edge'; break
        word = text[r['start']:r['end']]
        hidden = any(hx0 <= cx <= hx1 and hy0 <= cy <= hy1 for hx0, hy0, hx1, hy1 in features[k].get('hidden', ()))
        key = (k, round(x0 * 300), round(y0 * 300), word)
        duplicate = key in seen; seen.add(key)
        labels[i] = {'font': cls, 'region': region, 'region_id': region_id, 'text': word, 'hidden': hidden, 'duplicate': duplicate}
    # Ruled "tables" whose text is mostly in figure-style fonts (DejaVuSans, Helvetica, ...)
    # are charts with gridlines: treat them as figures.
    fonts_by_region = defaultdict(Counter)
    for lab in labels.values():
        if lab['region'] == 'table' and lab['region_id']: fonts_by_region[lab['region_id']][lab['font']] += 1
    # Tables often set their numbers in math mode, so only figure-style fonts count as chart evidence.
    chart = {rid for rid, c in fonts_by_region.items() if c['foreign'] >= .5 * sum(c.values())}
    for lab in labels.values():
        if lab['region_id'] in chart: lab['region'] = 'figure'
    return labels, body


WORD = re.compile(r"[A-Za-z][a-z'\-]{2,}[,.;:)]?")
REAL_WORD = re.compile(r"[A-Za-z][a-z'\-]+[,.;:)]?|[aAI][,.;:]?")      # body-font words never dropped for math
EQ_NUMBER = re.compile(r"\(\d+(\.\d+)?[a-z]?\)$")
LABEL_WORD = re.compile(r"^(Figure|Fig\.?|Table)$")
LABEL_NUM = re.compile(r"^[A-Z]?\d+(\.\d+)?[.:]?$")
GARBLED_CHAR = re.compile('[\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\ue000-\uf8ff\ufffd\u0000-\u0008]')
SECTION = re.compile(r"^([A-Z]|\d+)(\.\d+)*\.?$")
SHORT_IDENT = re.compile(r"^[(\[]?[A-Za-z]{1,4}[)\],.;:]?$")
NUMBER = re.compile(r"^[(\[]?([A-Za-z]{1,3}[=<>≤≥])?[−\-–+±~≈<>]?(\d[\d.,]*|\.\d+)(e[−\-+]?\d+)?%?(/[\d.,]+%?)*[)\],;:.]{0,2}$")
SIMPLE_MATH = re.compile(
    r"^[(\[]?[−\-–+±~≈<>]?\$?\d[\d.,]*(e[−\-+]?\d+)?[%×x]?([kKMBGT]|ms|s)?[)\],;:.]?$"      # numbers
    r"|^\d+(\.\d+)?[–\-]\d+(\.\d+)?[×x%]?[,.;:)]?$"                                        # ranges
    r"|^[(\[]?[A-Za-zα-ωΑ-Ωϵϑϕ][′']?[)\],.;:]?$"                                           # single variables
    r"|^[A-Za-zα-ω]{1,3}[=<>≤≥≈]\.?\d[\d.,]*%?[,.;:)]?$"                                    # p=.012, α=0.05
    r"|^[=<>≤≥≈∼±×·+\-−/:,()%|~]$")                                                         # lone operators


def classify(clean_result, labels):
    """Assign an omission type per output token of a cleaned result."""
    import bisect
    text = clean_result['text']; mapping = clean_result['mapping']
    starts = [0] + [m.end() for m in re.finditer('\n', text)]
    lines = defaultdict(list)
    for j, m in enumerate(mapping):
        lines[bisect.bisect_right(starts, m['start']) - 1].append(j)
    kind = [None] * len(mapping)
    lab = lambda j: labels.get(mapping[j]['word_index'], {})
    order = sorted(lines)
    # text boxes: drawn regions (figure or table) whose words sit mostly on prose lines
    # (prompt templates, theorem boxes, callouts, listings) stay text
    region_tokens = Counter(); region_prose = Counter(); line_prose = {}
    for ln in order:
        idx = lines[ln]
        line_prose[ln] = sum(1 for j in idx if lab(j).get('font') != 'math' and WORD.fullmatch(lab(j).get('text', ''))) >= 5
        for j in idx:
            rid = lab(j).get('region_id')
            if rid and lab(j).get('region') in ('figure', 'table'):
                region_tokens[rid] += 1; region_prose[rid] += line_prose[ln]
    region_numbers = Counter(); region_words = Counter(); region_kind = {}
    for j in range(len(mapping)):
        rid = lab(j).get('region_id')
        if rid and lab(j).get('region') in ('figure', 'table'):
            w = lab(j).get('text', ''); region_kind[rid] = lab(j).get('region')
            region_numbers[rid] += bool(NUMERIC.match(w)); region_words[rid] += bool(lab(j).get('font') != 'math' and REAL_WORD.fullmatch(w))
    text_boxes = {rid for rid, n in region_tokens.items() if n >= 20 and region_prose[rid] >= .5 * n}
    # theorem/algorithm boxes drawn as a frame: few numbers, a fair share of words, some prose
    text_boxes |= {rid for rid, n in region_tokens.items() if region_kind.get(rid) == 'table' and n >= 15
                   and region_numbers[rid] < .1 * n and region_words[rid] >= .3 * n and region_prose[rid] > 0}
    # never omitted: "Figure 3" / "Table 2" labels (also across a line break), the large first
    # capital of a small-caps word, and section headings ("4.2 Results", "D.4 QUESTION ANSWERING")
    protected = set()
    for j in range(len(mapping) - 1):
        a, b = lab(j).get('text', ''), lab(j + 1).get('text', '')
        if LABEL_WORD.match(a) and LABEL_NUM.match(b): protected |= {j, j + 1}
        if lab(j).get('font') == 'body' and re.fullmatch(r'[A-Z]', a) and re.match(r'^[\-–]?[A-Z]{2,}', b): protected.add(j)
    for ln in order:
        idx = lines[ln]; words = [lab(j).get('text', '') for j in idx]
        if len(idx) >= 2 and all(lab(j).get('font') == 'body' for j in idx) and SECTION.match(words[0]) \
                and sum(1 for w in words[1:] if re.fullmatch(r"[A-Za-z][A-Za-z'\-:]+[:.]?", w)) >= max(1, len(words) - 2):
            protected |= set(idx)
    caption_mode = False
    for ln in order:
        idx = lines[ln]
        for j in idx:   # invisible or doubled text is dropped silently, whatever else it is
            if lab(j).get('hidden'): kind[j] = 'hidden'
            elif lab(j).get('duplicate'): kind[j] = 'duplicate'
        toks = [(j, lab(j)) for j in idx if kind[j] is None]
        if not toks: continue
        line_text = text[mapping[idx[0]]['start']:mapping[idx[-1]]['end']]
        body_words = sum(1 for _, t in toks if t.get('font') == 'body' and WORD.fullmatch(t.get('text', '')))
        regions = Counter(t.get('region') for _, t in toks)
        if CAPTION.match(line_text) or any(j in protected for j, _ in toks[:2]): caption_mode = True
        elif caption_mode and (body_words < max(2, .5 * len(toks)) or regions.get('table')): caption_mode = False
        if caption_mode: continue
        prose_line = body_words >= 5
        # a full prose line inside a table region is text set beside or between tables
        prose_heavy = prose_line and body_words >= .6 * len(toks)
        table_line = regions.get('table', 0) >= .5 * len(toks) and not prose_heavy
        for j, t in toks:
            if j in protected: continue
            if t.get('region') == 'table' and t.get('region_id') in text_boxes: continue
            if table_line or (t.get('region') == 'table' and not (prose_heavy and t.get('font') == 'body')): kind[j] = 'table'; continue
            if t.get('region') in ('figure', 'figure_edge') and t.get('region_id') not in text_boxes:
                # body-font words in a prose-like line are wrapped text beside a figure, not figure text
                if not (prose_line and t.get('font') == 'body'): kind[j] = 'figure'
        rest = [(j, t) for j, t in toks if kind[j] is None and j not in protected]
        if not rest: continue
        math_toks = sum(1 for _, t in rest if t.get('font') == 'math')
        chars = sum(len(t.get('text', '')) for _, t in rest) or 1
        math_chars = sum(len(t.get('text', '')) for _, t in rest if t.get('font') == 'math')
        real_words = sum(1 for _, t in toks if t.get('font') == 'body' and REAL_WORD.fullmatch(t.get('text', '')))
        display = math_chars / chars >= .5 and (math_toks >= 4 or EQ_NUMBER.search(line_text.strip())) and real_words <= 1
        for j, t in rest:
            if t.get('font') == 'body' and REAL_WORD.fullmatch(t.get('text', '')): continue   # never drop real words for math
            if display: kind[j] = 'math_display'
            elif t.get('font') == 'math': kind[j] = 'math_inline'
        # absorb short symbol tokens sitting between inline math tokens
        for a in range(1, len(idx) - 1):
            j = idx[a]
            if kind[j] is None and kind[idx[a - 1]] == 'math_inline' and kind[idx[a + 1]] == 'math_inline' \
                    and re.fullmatch(r"[^A-Za-z]{1,3}", lab(j).get('text', 'xx')):
                kind[j] = 'math_inline'
        _keep_simple_math(idx, lab, kind)
    _merge_equations(order, lines, mapping, lab, kind)
    _keep_readable_math(order, lines, lab, kind)
    _caption_tables(order, lines, mapping, lab, kind, text)
    _numeric_runs(lines, mapping, labels, kind)
    for j in protected:   # later passes must not take labels or headings
        if kind[j] not in ('hidden', 'duplicate'): kind[j] = None
    _references(order, lines, mapping, kind, text)
    return kind


REF_HEADING = {'references', 'bibliography', 'literaturecited', 'workscited', 'referencesandnotes'}
APPENDIX_HEADING = re.compile(r'^(Appendix|APPENDIX|A\s?PPENDIX|Appendices|APPENDICES|Supplementary|SUPPLEMENTARY|S\s?UPPLEMENTARY|'
                              r'(NeurIPS )?Paper Checklist|Checklist|CHECKLIST)\b|^[A-Z](\.\d+)*\s+[A-Z]')


REF_SIGNAL = re.compile(r'(19|20)\d\d|et al|arXiv|https?://|doi|Proceedings|Conference|Journal|Transactions|pp\.|[Vv]ol\.|URL|Press|Workshop|Advances in|preprint|\bIn\b')


def _references(order, lines, mapping, kind, text):
    """The bibliography is not author prose to classify: every token from a References /
    Bibliography heading to the end of the reference list is omitted. The list ends at an
    appendix or section heading (same line or a lone letter on its own line), an all-caps
    heading, or after 8 lines in a row with no sign of a reference entry."""
    def line_text(ln):
        idx = lines[ln]; return text[mapping[idx[0]]['start']:mapping[idx[-1]]['end']].strip()
    def heading(k):
        t = line_text(order[k]); n = len(lines[order[k]])
        if n > 12 or re.search(r'(19|20)\d\d|et al|[.,;]$', t): return False
        if APPENDIX_HEADING.match(t) or re.fullmatch(r'[A-Z](\.\d+)*', t): return True
        letters = re.sub(r'[^A-Za-z]', '', t)
        return n <= 8 and len(letters) >= 4 and letters.isupper()
    k = 0
    while k < len(order):
        bare = re.sub(r'^\d+(\.\d+)*\.?\s*', '', line_text(order[k]))
        if re.sub(r'\s+', '', bare).lower() not in REF_HEADING: k += 1; continue
        end = len(order); last = k; gap = 0
        for q in range(k + 1, len(order)):
            if heading(q): end = q; break
            if REF_SIGNAL.search(line_text(order[q])): last = q; gap = 0
            else:
                gap += 1
                if gap >= 8: end = last + 1; break
        for q in range(k, end):
            for j in lines[order[q]]: kind[j] = 'references'
        k = max(end, k + 1)


def _keep_simple_math(idx, lab, kind):
    """Inline math that reads as plain text (numbers, ranges, B = 512, p=.012, c_puct = 1.0)
    stays text; numbers inside longer inline math are kept too, since they carry results."""
    a = 0
    while a < len(idx):
        if kind[idx[a]] != 'math_inline': a += 1; continue
        b = a
        while b + 1 < len(idx) and kind[idx[b + 1]] == 'math_inline': b += 1
        run = [lab(idx[x]).get('text', '') for x in range(a, b + 1)]
        simple = all(SIMPLE_MATH.match(w) or SHORT_IDENT.match(w) for w in run)
        if simple and ((len(run) <= 9 and any(c.isdigit() for w in run for c in w)) or len(run) <= 2):
            for x in range(a, b + 1): kind[idx[x]] = None
        else:
            for x in range(a, b + 1):
                if NUMBER.match(run[x - a]): kind[idx[x]] = None
        a = b + 1


def _garbled(word):
    """A math-font token whose text the PDF mis-encoded: accented Latin letters standing in
    for symbols (ď for ≤), private-use glyphs, replacement characters, or the p...q / r...s
    pattern that some fonts emit for (...) and [...]."""
    return bool(GARBLED_CHAR.search(word) or (('(' not in word and re.search(r'p[^\s()]{1,12}q', word))
                                               or ('[' not in word and re.fullmatch(r'r[^\s\[\]]{1,12}s[,.;:]?', word))))


def _keep_readable_math(order, lines, lab, kind):
    """Policy (2026-10-06): inline math that reads correctly stays text; only garbled math
    becomes a marker. A run of inline-math tokens on a line with real words is kept unless
    any token in it is garbled. Math-only lines (equation fragments, scattered sub- and
    superscripts) stay omitted."""
    for ln in order:
        idx = lines[ln]
        if sum(1 for j in idx if lab(j).get('font') == 'body' and REAL_WORD.fullmatch(lab(j).get('text', ''))) < 2: continue
        a = 0
        while a < len(idx):
            if kind[idx[a]] != 'math_inline': a += 1; continue
            b = a
            while b + 1 < len(idx) and kind[idx[b + 1]] == 'math_inline': b += 1
            run = [lab(idx[x]).get('text', '') for x in range(a, b + 1)]
            if not any(_garbled(w) for w in run):
                for x in range(a, b + 1): kind[idx[x]] = None
            a = b + 1


def _merge_equations(order, lines, mapping, lab, kind):
    """One display equation = one marker: absorb inline-math fragments, equation numbers and
    pure-math lines adjacent to a display equation on the same page."""
    def mathish(ln):
        toks = [j for j in lines[ln] if kind[j] not in ('figure', 'table', 'hidden', 'duplicate')]
        return toks and all(kind[j] in ('math_inline', 'math_display') or EQ_NUMBER.fullmatch(lab(j).get('text', '').strip())
                            or re.fullmatch(r"[^A-Za-z]{1,3}", lab(j).get('text', 'xx')) for j in toks)
    for n, ln in enumerate(order):
        if not any(kind[j] == 'math_display' for j in lines[ln]): continue
        group = [ln]
        for step in (-1, 1):
            m = n + step
            while 0 <= m < len(order) and mapping[lines[order[m]][0]]['page'] == mapping[lines[ln][0]]['page'] and mathish(order[m]):
                group.append(order[m]); m += step
        for g in group:
            for j in lines[g]:
                if kind[j] in ('math_inline', None) and (kind[j] == 'math_inline' or EQ_NUMBER.fullmatch(lab(j).get('text', '').strip())
                                                         or re.fullmatch(r"[^A-Za-z]{1,3}", lab(j).get('text', 'xx'))):
                    kind[j] = 'math_display'


def _caption_tables(order, lines, mapping, lab, kind, text):
    """Tables without drawn rules: the non-prose block right after (or before) a "Table N"
    caption, if it is number-heavy, is a table."""
    TABLE_CAP = re.compile(r"^\s*Table\s*[A-Z]?\d+")
    def block(n, step):
        out = []; m = n + step; page = mapping[lines[order[n]][0]]['page']
        while 0 <= m < len(order) and len(out) < 80:
            idx = lines[order[m]]
            if mapping[idx[0]]['page'] != page: break
            line_text = text[mapping[idx[0]]['start']:mapping[idx[-1]]['end']]
            body_words = sum(1 for j in idx if lab(j).get('font') == 'body' and WORD.fullmatch(lab(j).get('text', '')))
            if body_words >= 5 or CAPTION.match(line_text): break
            out.append(order[m]); m += step
        return out
    for n, ln in enumerate(order):
        idx = lines[ln]
        if not TABLE_CAP.match(text[mapping[idx[0]]['start']:mapping[idx[-1]]['end']]): continue
        for step in (1, -1):
            blk = block(n, step)
            toks = [j for g in blk for j in lines[g] if kind[j] is None]
            if len(blk) >= 3 and toks and sum(1 for j in toks if NUMERIC.match(lab(j).get('text', ''))) >= .25 * len(toks):
                for j in toks: kind[j] = 'table'
                break


def _numeric_runs(lines, mapping, labels, kind):
    """Omit runs of number-only text (axis ticks, legend values) outside tables:
    3+ consecutive lines of only numbers, or one line of 6+ numbers and no words."""
    def numeric_line(idx):
        toks = [j for j in idx if kind[j] is None]
        if not toks: return None
        words = [labels.get(mapping[j]['word_index'], {}).get('text', '') for j in toks]
        nums = sum(1 for w in words if NUMERIC.match(w))
        if nums and all(NUMERIC.match(w) or len(w) <= 2 for w in words): return toks, nums
        return None
    run = []
    def flush():
        n_lines = len(run); n_nums = sum(r[1] for r in run)
        if (n_lines >= 3 and n_nums >= 4) or (n_lines >= 1 and max(r[1] for r in run) >= 6):
            for toks, *_ in run:
                for j in toks: kind[j] = 'numeric'
    for ln in sorted(lines):
        idx = lines[ln]
        page = mapping[idx[0]]['page']
        r = numeric_line(idx)
        if r and (not run or run[-1][2] == page):
            run.append((r[0], r[1], page)); continue
        if run: flush(); run = []
        if r: run.append((r[0], r[1], page))
    if run: flush()


def render(clean_result, kinds, policy=DEFAULT_POLICY):
    """Replace runs of omitted tokens with markers. Returns text and omission records."""
    text = clean_result['text']; mapping = clean_result['mapping']
    out = []; omissions = []; pos = 0; j = 0
    while j < len(mapping):
        k = kinds[j]
        if k and policy.get(k):
            a = j
            while j + 1 < len(mapping) and kinds[j + 1] == k and mapping[j + 1]['page'] == mapping[a]['page']: j += 1
            # one marker per run; display/figure/table runs sit on their own line
            out.append(text[pos:mapping[a]['start']])
            marker = MARKERS[k]; block = k != 'math_inline' and bool(marker)
            prefix = '' if not block or not out[-1] or out[-1].endswith('\n') else '\n'
            last = text[mapping[j]['start']:mapping[j]['end']]
            tail = last[-1] if marker and not block and last[-1:] in '.,;:' and len(last) > 1 else ''
            out.append(prefix + marker + tail + ('\n' if block else ''))
            omissions.append({'type': k, 'words': [mapping[x]['word_index'] for x in range(a, j + 1)],
                              'source_start': mapping[a]['source_start'], 'source_end': mapping[j]['source_end'],
                              'page': mapping[a]['page'], 'text': text[mapping[a]['start']:mapping[j]['end']][:400]})
            pos = mapping[j]['end']
            if block:
                while pos < len(text) and text[pos] in ' \n': pos += 1
        j += 1
    out.append(text[pos:])
    # Merge back-to-back identical markers, but never across a page break (\f).
    result = re.sub(r'(⟦[a-z ]+⟧)([ \n]*\1)+', r'\1', ''.join(out))
    return re.sub(r'\n{3,}', '\n\n', result), omissions
