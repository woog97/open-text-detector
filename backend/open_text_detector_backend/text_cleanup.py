"""Conservative derived PDF text. Never changes the canonical artifact.

Only positively identified margin boilerplate is dropped. Ambiguous words/math
are retained. Source mappings are token-granular for transformed words; copied
words support exact character projection. No semantic labels are inferred.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata

VERSION = 'positioned-clean-v2'
LIGATURES = {'ﬀ':'ff', 'ﬁ':'fi', 'ﬂ':'fl', 'ﬃ':'ffi', 'ﬄ':'ffl', 'ﬅ':'st', 'ﬆ':'st'}
# Adobe Symbol private-use extension pieces of large delimiters (Adobe Glyph List).
SYMBOL_PIECES = dict(zip(range(0xF8EB, 0xF8FF), '⎛⎜⎝⎡⎢⎣⎧⎨⎩⎪⎮⎞⎟⎠⎤⎥⎦⎫⎬⎭'))
CHAR_MAP = str.maketrans({**LIGATURES, **{chr(k):v for k,v in SYMBOL_PIECES.items()}})
# Variation selectors, zero-width/direction marks, soft hyphen, controls and
# U+FFFD (glyphs without a Unicode mapping) carry no recoverable text.
INVISIBLE = re.compile('[\ufe00-\ufe0f\u200b-\u200f\u2060\ufeff\u00ad\x00-\x1f\x7f]')
NEGATION = '\u0338'
HEADINGS = {'ABSTRACT', 'INTRODUCTION', 'REFERENCES', 'CONCLUSION', 'CONCLUSIONS',
            'ACKNOWLEDGMENTS', 'ACKNOWLEDGEMENTS', 'APPENDIX', 'APPENDICES',
            'DISCUSSION', 'LIMITATIONS', 'BACKGROUND', 'EXPERIMENTS'}
HEADER = re.compile(r'^(?:(?:Under review as a conference paper|Published as a conference paper) at (?:ICLR|COLM) 20\d\d\.?|'
                    r'Published in Transactions on Machine Learning Research \((?:\d{1,2}|[A-Z][a-z]+)/20\d\d\)|'
                    r'Under review as submission to TMLR)$', re.I)
CHECKLIST_FILE = Path(__file__).with_name('checklist_template.json')
# Document-level evidence of PDF fonts whose ToUnicode maps are wrong
# (− as ´, = as “, + as `); the text cannot be repaired without font data.
BROKEN_MATH_FONT = re.compile(r'[A-Za-z0-9)]´\d|\b[a-z]“\d|\b[a-z]`1\b')
LATEX_SOURCE = re.compile(r'\\(?:hat|mathbf|mathcal|mathrm|frac|textbf|ensuremath|protect|left|right|begin)\b')
# Productive prefixes are too ambiguous to remove their hyphen automatically.
COMPOUND_PREFIXES = {'non','pre','post','re','co','multi','cross','self','semi','anti',
                    'inter','intra','high','low','long','short','out','in','on','off',
                    'well','state','agent','world','task','model','data','domain'}

@dataclass
class Line:
    words: list
    original: int
    page: int

    @property
    def x0(self): return min(w['x0'] for w in self.words)
    @property
    def x1(self): return max(w['x1'] for w in self.words)
    @property
    def y0(self): return min(w['y0'] for w in self.words)
    @property
    def y1(self): return max(w['y1'] for w in self.words)
    @property
    def height(self): return statistics.median(w['y1']-w['y0'] for w in self.words)
    @property
    def text(self): return ' '.join(w['value'] for w in self.words)


def _negated(char):
    composed=unicodedata.normalize('NFC',char+NEGATION)
    return composed if len(composed)==1 else None


def _normalize(value, strip_negation=False, negate_first=False):
    """Deterministic per-token character repair; validate() recomputes it."""
    if strip_negation:value=value[:-1]
    if negate_first:value=_negated(value[0])+value[1:]
    value=INVISIBLE.sub('',value.translate(CHAR_MAP)).replace('�','')
    value=re.sub(NEGATION+'(.)',lambda m:_negated(m.group(1)) or m.group(0),value)
    return unicodedata.normalize('NFC',value)


def _token_repairs(artifact):
    """Normalize every source word. Poppler emits a negation slash before the
    negated symbol, often as its own word ('̸ =' for ≠); compose it forward."""
    text=artifact['text'];boxes=artifact['rectangles'];ops=[{} for _ in boxes]
    for i,box in enumerate(boxes[:-1]):
        value=text[box['start']:box['end']];nxt=boxes[i+1]
        following=text[nxt['start']:nxt['end']]
        if (value.endswith(NEGATION) and nxt['page']==box['page'] and
                '\n' not in text[box['end']:nxt['start']] and _negated(following[0])):
            ops[i]['strip_negation']=True;ops[i+1]['negate_first']=True
    values=[_normalize(text[b['start']:b['end']],**op) for b,op in zip(boxes,ops)]
    return values,ops


def _norm_line(line): return ' '.join(line.split())
def _hash(value): return hashlib.sha256(value.encode()).hexdigest()[:16]
def _shingles(line):
    words=line.lower().split()
    if len(re.findall(r'[A-Za-z]{2,}',line))<4 or len(words)<5:return []
    return [_hash(' '.join(words[i:i+4])) for i in range(len(words)-3)]


@lru_cache(maxsize=1)
def _checklist_template():
    data=json.loads(CHECKLIST_FILE.read_text())
    return frozenset(data['shingles']),frozenset(data['short_lines'])


def _checklist(lines):
    """NeurIPS checklist questions/guidelines are conference template text, not
    author prose. Remove only fingerprinted template lines, and only when a paper
    contains many of them; Answer/Justification lines are always kept."""
    shingles,short=_checklist_template();long_lines=[];short_lines=[]
    for line in lines:
        text=_norm_line(line.text)
        if re.match(r'(Answer|Justification)\b',text):continue
        s=_shingles(text)
        if s and sum(h in shingles for h in s)>=.8*len(s):long_lines.append(line)
        elif _hash(text) in short:short_lines.append(line)
    if len({_norm_line(l.text) for l in long_lines})<20:return {}
    return {w['index']:'checklist_template' for line in long_lines+short_lines for w in line.words}


def _split_letter_runs(line):
    """Word indices (j) whose preceding separator is not a space: letters that
    the PDF positions individually, horizontally (tiny figure text, code),
    letter-spaced capitals (tracked headings) or stacked bottom-up (rotated axis
    labels). Requires >=4-token runs of same-size glyphs so ordinary prose and
    math are never joined; a gap well above the run's letter spacing stays a space."""
    words=line.words;joined=set()
    def h(w):return max(.1,w['y1']-w['y0'])
    def width(w):return w['x1']-w['x0']
    def hgap(a,b):return b['x0']-a['x1']
    def vgap(a,b):return a['y0']-b['y1']  # positive: b sits above a (reads bottom-up)
    def aligned(a,b):return .9<=h(a)/h(b)<=1.11 and abs(a['y1']-b['y1'])<=.1*min(h(a),h(b))
    def split(a,b):
        return (re.fullmatch('[A-Za-z_]{1,3}',a['value']) and re.fullmatch('[A-Za-z_]{1,3}',b['value']) and
                aligned(a,b) and -.2<=hgap(a,b)<=.17*min(h(a),h(b)))
    def tracked(a,b):
        return (re.fullmatch(r'[A-Z/.:_\-]',a['value']) and re.fullmatch(r'[A-Z/.:_\-]',b['value']) and
                aligned(a,b) and -.2<=hgap(a,b)<=.35*min(h(a),h(b)))
    def vertical(a,b):
        return (len(a['value'])==1 and len(b['value'])==1 and width(a)>0 and
                abs(a['x0']-b['x0'])<=.3 and abs(a['x1']-b['x1'])<=.3 and -.3<=vgap(a,b)<=.5*width(a))
    for linked,gap,size in ((split,hgap,h),(tracked,hgap,h),(vertical,vgap,width)):
        run=[0]
        for j in range(1,len(words)+1):
            if j<len(words) and linked(words[j-1],words[j]):run.append(j);continue
            tokens=[words[k]['value'] for k in run]
            if (len(run)>=4 and sum(map(len,tokens))/len(run)<=2.5 and
                    (linked is not tracked or sum(t.isalpha() for t in tokens)>=3)):
                gaps=[gap(words[k-1],words[k]) for k in run[1:]];typical=statistics.median(gaps)
                for k,g in zip(run[1:],gaps):
                    if g<=max(1.6*typical,typical+.06*size(words[k])):joined.add(k)
            run=[j]
    return joined


def _lines(artifact):
    """Retain original line boundaries; split large gaps or tiny diagram runs."""
    text = artifact['text']; lines = []; current = []; previous = None
    for i, box in enumerate(artifact['rectangles']):
        word = {**box, 'index': i, 'value': text[box['start']:box['end']]}
        if previous is not None:
            gap = text[previous['end']:box['start']]
            split = ('\n' in gap or '\f' in gap or box['page'] != previous['page'])
            if split and current:
                lines.append(Line(current, len(lines), current[0]['page'])); current=[]
        current.append(word); previous=box
    if current: lines.append(Line(current, len(lines), current[0]['page']))
    separated=[]
    def lexical(words):
        return len(words)>=4 and sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z’'\-,.;:()]*",w['value'])) for w in words)/len(words)>=.65
    for line in lines:
        start=0
        for j in range(1,len(line.words)):
            a,b=line.words[j-1:j+1]
            ah=max(1.,a['y1']-a['y0']);bh=max(1.,b['y1']-b['y0'])
            if (min(ah,bh)/max(ah,bh)<.7 and min(ah,bh)<=6.5 and max(ah,bh)>=7.5 and
                b['x0']-a['x1']>max(4.,.6*min(ah,bh)) and
                lexical(line.words[start:j]) and lexical(line.words[j:])):
                separated.append(Line(line.words[start:j],len(separated),line.page));start=j
        separated.append(Line(line.words[start:],len(separated),line.page))
    return separated


def _boilerplate(lines, pages):
    removed = {}; header_pages = defaultdict(set)
    for line in lines:
        if line.y1 < pages[line.page]['height']*.085 and HEADER.fullmatch(line.text):
            header_pages[line.text].add(line.page)
    for line in lines:
        if (line.y1 < pages[line.page]['height']*.085 and
                len(header_pages[line.text])>=2):
            for w in line.words: removed[w['index']] = 'repeated_review_header'
    # Double-blind author placeholder: the exact template pair, near the top of page 1.
    top=[l for l in lines if l.page==1 and l.y1<pages[1]['height']*.35]
    for a,b in zip(top,top[1:]):
        if a.text=='Anonymous authors' and b.text=='Paper under double-blind review':
            for w in a.words+b.words:removed[w['index']]='anonymous_author_placeholder'
    # Infer each numeric margin rail from a long, evenly spaced consecutive run.
    # Restrict to outside the central 70% of the page; table numbers are untouched.
    # Candidates are words, not lines: poppler sometimes merges a margin number
    # into the adjacent body line.
    rails = defaultdict(list)
    for line in lines:
        p=pages[line.page]
        for w in line.words:
            side=('left' if w['x1'] < p['width']*.16 else 'right' if w['x0'] > p['width']*.84 else None)
            if (side and re.fullmatch(r'\d{3,4}',w['value']) and
                    p['height']*.08 < w['y0'] < p['height']*.94):
                rails[(line.page, side)].append(Line([w], line.original, line.page))
    accepted=defaultdict(list)
    for (page,side),group in sorted(rails.items()):
        group.sort(key=lambda l:l.y0)
        runs=[]; run=[]
        for line in group:
            if run and (int(line.text)!=int(run[-1].text)+1 or
                        not .7*run[-1].height < line.y0-run[-1].y0 < 2.5*run[-1].height):
                runs.append(run);run=[]
            run.append(line)
        runs.append(run)
        for run in runs:
            if len(run)<12: continue
            deltas=[b.y0-a.y0 for a,b in zip(run,run[1:])]
            if max(deltas)-min(deltas)>1.5: continue
            # ICLR/ICML review templates number 54 lines/page, globally from zero.
            # Accept a run that matches that, continues the previous accepted run
            # (numbering offset by unnumbered pages), or restarts/jumps with a
            # near-full-page run (appended supplements). Repeated per-page table
            # labels fail all three.
            first=int(run[0].text);previous=accepted[side][-1] if accepted[side] else None
            matches_page=all(54*(page-1)<=int(l.text)<54*page for l in run)
            continues=previous is not None and first==int(previous[-1].text)+1
            if matches_page or continues or len(run)>=40:accepted[side].append(run)
    for runs in accepted.values():
        if len({r[0].page for r in runs})<2:continue
        for run in runs:
            for line in run:removed[line.words[0]['index']]='review_line_number'
    # Page numbers must equal the physical page index, be isolated and bottom-centred,
    # and appear on at least two pages. Never remove body numerals or equation labels.
    footers=[]
    for line in lines:
        p=pages[line.page]
        if (len(line.words)==1 and line.text==str(line.page) and
            line.y0>p['height']*.93 and abs((line.x0+line.x1)/2-p['width']/2)<p['width']*.08):
            footers.append(line)
    if len({l.page for l in footers})>=2:
        for line in footers: removed[line.words[0]['index']]='page_number'
    return removed


def _prose(line):
    words=line.text.split()
    alpha=sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z’'\-,.;:()]*",w)) for w in words)
    return len(words)>=4 and alpha/len(words)>=.65 and 7<=line.height<=16


def _lanes(lines):
    """Repair only strong, vertically continuous prose lanes interrupted by other regions.

    Other regions must be geometrically disjoint. If anything overlaps the lane,
    leave the order intact and flag it for review. No page-wide coordinate sort.
    """
    groups=defaultdict(list)
    for line in lines:
        if _prose(line): groups[(round(line.x0/3),round(line.height))].append(line)
    candidates=[]
    for group in groups.values():
        group.sort(key=lambda l:l.y0); run=[]
        for line in group+[None]:
            if run and (line is None or not .65*run[-1].height < line.y0-run[-1].y0 < 1.9*run[-1].height or
                           abs(line.x1-run[-1].x1)>max(20., .1*(run[-1].x1-run[-1].x0))):
                if len(run)>=5 and sum(len(l.text) for l in run)>=180:
                    candidates.append(run)
                run=[]
            if line is not None:run.append(line)
    ordered=list(lines); events=[]; flagged=[]
    for run in sorted(candidates,key=lambda r:-len(r)):
        ids={l.original for l in run};positions=[i for i,l in enumerate(ordered) if l.original in ids]
        if len(positions)!=len(run):continue
        lo,hi=min(positions),max(positions)
        intruders=[l for l in ordered[lo:hi+1] if l.original not in ids]
        if not intruders:continue
        # Repair side-by-side regions only. Different prose columns remain independent.
        x0=min(l.x0 for l in run);x1=max(l.x1 for l in run)
        if any(not (l.x1 < x0-5 or l.x0 > x1+5) for l in intruders):
            flagged.append({'kind':'ambiguous_reading_order','page':run[0].page,
                            'source_start':run[0].words[0]['start']});continue
        # Require visual order to agree with the existing relative prose order.
        if [l.original for l in ordered if l.original in ids] != [l.original for l in run]:continue
        ordered[lo:hi+1]=run+intruders
        events.append({'kind':'separated_prose_lane','page':run[0].page,
                       'lines':len(run),'source_start':run[0].words[0]['start'],
                       'body_lines':[l.original for l in run],
                       'side_lines':[l.original for l in intruders]})
    return ordered,events,flagged


def _title_order(lines):
    # A displaced first-page title is only moved when it is above every other
    # non-header/non-margin content line. This excludes section/figure labels.
    if not lines or lines[0].page!=1:return lines,[]
    candidates=[l for l in lines if l.y0<200 and l.height>=12 and
                len(re.findall('[A-Z]',l.text))>=8 and
                len(re.findall('[a-z]',l.text))==0]
    if not candidates:return lines,[]
    candidates.sort(key=lambda l:l.y0); title=[candidates[0]]
    for l in candidates[1:]:
        if 0 < l.y0-title[-1].y0 < 2*title[-1].height:title.append(l)
        else:break
    ids={l.original for l in title};rest=[l for l in lines if l.original not in ids]
    if not rest or min(l.y0 for l in rest)<max(l.y1 for l in title)-1:return lines,[]
    if [l.original for l in lines[:len(title)]]==[l.original for l in title]:return lines,[]
    return title+rest,[{'kind':'restored_top_title','page':1,'lines':len(title)}]


def _heading_joins(line):
    vals=[w['value'] for w in line.words];start=0
    if ''.join(vals) in HEADINGS:return set(range(1,len(vals)))
    if vals and re.fullmatch(r'\d+(?:\.\d+)*|[A-Z]',vals[0]):start=1
    if ''.join(vals[start:]) not in HEADINGS:return set()
    return set(range(start+1,len(vals)))


def _continuous(a,b):
    # Only full aligned prose lines within a page; paragraph indent/vertical gap
    # and math/table/code layouts retain their line breaks.
    return (a.page==b.page and _prose(a) and _prose(b) and
            abs(a.x0-b.x0)<=2 and abs(a.height-b.height)<=1 and
            .75*a.height < b.y0-a.y0 < 1.65*a.height and
            a.x1-b.x0>140 and len(a.text)>35 and
            not re.match(r'^(?:Figure|Table|Algorithm)\s+\d',b.text))


FIGURE_BODY_TYPES = {'chart_body', 'image_body'}


def mineru_figure_regions(middle_json):
    """Figure/image body boxes from a MinerU layout JSON, as {page (1-based): [[x0,y0,x1,y1], ...]}
    in page fractions. Captions and footnotes are separate blocks and are not included."""
    regions = defaultdict(list)
    def walk(node, page):
        if isinstance(node, dict):
            if node.get('type') in FIGURE_BODY_TYPES and len(node.get('bbox') or []) == 4:
                regions[page].append([float(v) for v in node['bbox']])
            for value in node.values(): walk(value, page)
        elif isinstance(node, list):
            for value in node: walk(value, page)
    for page in middle_json.get('pages', []):
        walk(page.get('blocks', []), page['page_idx'] + 1)
    return dict(regions)


def _figure_words(lines, pages, regions, margin=.003):
    """Words whose centre lies inside a figure body: axis ticks, legends, diagram labels."""
    removed = {}
    for line in lines:
        boxes = regions.get(line.page)
        if not boxes: continue
        p = pages[line.page]
        for w in line.words:
            cx = (w['x0'] + w['x1']) / 2 / p['width']; cy = (w['y0'] + w['y1']) / 2 / p['height']
            if any(x0 - margin <= cx <= x1 + margin and y0 - margin <= cy <= y1 + margin for x0, y0, x1, y1 in boxes):
                removed[w['index']] = 'figure_region'
    return removed


def clean(artifact, figure_regions=None):
    """figure_regions: optional output of mineru_figure_regions(); words inside those
    figure bodies are removed (recorded as 'figure_region') so prose excludes figure text."""
    text=artifact['text'];source_hash=hashlib.sha256(text.encode()).hexdigest()
    if source_hash!=artifact['text_sha256']:raise ValueError('Canonical text hash mismatch')
    previous=0
    for box in artifact['rectangles']:
        if not previous<=box['start']<box['end']<=len(text):raise ValueError('Invalid canonical offsets')
        if text[previous:box['start']].strip():raise ValueError('Unmapped canonical content')
        if not all(math.isfinite(box[k]) for k in ['x0','y0','x1','y1']):raise ValueError('Nonfinite geometry')
        previous=box['end']
    if text[previous:].strip():raise ValueError('Unmapped canonical tail')
    pages={p['page']:p for p in artifact['pages']};lines=_lines(artifact)
    removed=_boilerplate(lines,pages)
    for i,reason in _checklist(lines).items():removed.setdefault(i,reason)
    if figure_regions:
        for i,reason in _figure_words(lines,pages,figure_regions).items():removed.setdefault(i,reason)
    values,ops=_token_repairs(artifact)
    for i,(box,value) in enumerate(zip(artifact['rectangles'],values)):
        if value or i in removed:continue
        raw=text[box['start']:box['end']]
        removed[i]=('combined_negation' if ops[i].get('strip_negation') else
                    'unmapped_glyph' if '�' in raw else 'invisible_character')
    bypage=defaultdict(list)
    for line in lines:
        kept=[w for w in line.words if w['index'] not in removed]
        if kept:bypage[line.page].append(Line(kept,line.original,line.page))
    ordered=[];events=[];flags=[]
    for p in artifact['pages']:
        ll,ev,fl=_lanes(bypage[p['page']]);ll,te=_title_order(ll)
        ordered+=ll;events+=ev+te;flags+=fl
    layout_regions={}
    for n,event in enumerate(events):
        if event['kind']=='separated_prose_lane':
            for i in event['body_lines']:layout_regions[i]=f'body-lane-{n}'
            for i in event['side_lines']:layout_regions[i]=f'side-region-{n}'
    vocabulary=Counter(re.findall(r'\b[A-Za-z]{4,}\b',text))
    hyphenated=set(re.findall(r'\b[A-Za-z]+-[A-Za-z]+\b',text))
    chunks=[];mapping=[];regions=[];offset=0;seen=[];prior=None
    def emit(value,word=None,kind='copy'):
        nonlocal offset
        if not value:return
        chunks.append(value)
        if word is not None:
            mapping.append({'start':offset,'end':offset+len(value),'source_start':word['start'],
                            'source_end':word['end'],'word_index':word['index'],'page':word['page'],'kind':kind,
                            **ops[word['index']]})
        offset+=len(value)
    for line in ordered:
        joins=_heading_joins(line);continuous=prior is not None and _continuous(prior,line)
        if prior is not None and layout_regions.get(prior.original)!=layout_regions.get(line.original):continuous=False
        sep='' if prior is None else ('\n\f\n' if prior.page!=line.page else (' ' if continuous else '\n'))
        if prior is not None and prior.page==line.page and layout_regions.get(prior.original)!=layout_regions.get(line.original):sep='\n\n'
        # Dehyphenate only with repeated intact evidence in this same document.
        # This sacrifices recall to protect valid compounds and mathematical minus signs.
        if continuous and prior.words and line.words:
            left=prior.words[-1]['value'];right=line.words[0]['value'];joined=left[:-1]+right
            if (re.fullmatch('[A-Za-z]{2,}-',left) and re.fullmatch('[a-z]{2,}[,.;:]?',right) and
                left[:-1].lower() not in COMPOUND_PREFIXES and
                vocabulary[joined.rstrip(',.;:')]>=2 and left+right not in hyphenated and
                chunks and chunks[-1].endswith('-')):
                chunks[-1]=chunks[-1][:-1];offset-=1;mapping[-1]['end']-=1
                mapping[-1]['source_end']-=1;sep=''
                events.append({'kind':'dehyphenated','source_start':prior.words[-1]['end']-1,
                               'source_end':prior.words[-1]['end'],'page':line.page})
        if continuous and prior.words[-1]['value'].endswith('-') and sep:
            compound=prior.words[-1]['value']+line.words[0]['value']
            if compound in hyphenated:
                sep=''
                events.append({'kind':'joined_hyphenated_wrap','source_start':prior.words[-1]['end'],'page':line.page})
            else:
                sep='\n'
                flags.append({'kind':'ambiguous_hyphenation','source_start':prior.words[-1]['start'],'page':line.page})
        emit(sep);start=offset;letters=_split_letter_runs(line)
        heading_style=(len(re.findall('[A-Z]',line.text))>=8 and
                       re.fullmatch(r"[A-Z0-9\s:(),.'’\-–—?!/&+]+",line.text) is not None)
        for j,w in enumerate(line.words):
            if j:
                # Explicit known headings OR nearly touching small-cap typographic runs.
                prev=line.words[j-1];tight=(w['x0']-prev['x1'])
                smallcap=(heading_style and 0<=tight<=.12*line.height and
                          abs(w['y1']-prev['y1'])<=max(1.,.1*line.height))
                # In running text: a capital touching a shorter all-caps word on the
                # same baseline (D IST -> DIST). Subscripts are smaller and shifted.
                ph=max(.1,prev['y1']-prev['y0']);wh=w['y1']-w['y0']
                inline_smallcap=(re.fullmatch('[^A-Za-z0-9]*[A-Z]',prev['value']) and re.fullmatch(r'[A-Z]+',w['value']) and
                                 -.1*ph<=tight<=.12*ph and .7<=wh/ph<=.92 and abs(w['y1']-prev['y1'])<=.06*ph)
                # Overlapping all-caps pieces of one compound (C LIENT –E DGE -> CLIENT–EDGE).
                inline_smallcap=inline_smallcap or (heading_style and re.search('[A-Z]',prev['value']) and
                                 re.search('[A-Z]',w['value']) and -.1*max(ph,wh)<=tight<=0 and abs(w['y1']-prev['y1'])<=.06*max(ph,wh))
                if j in letters:
                    events.append({'kind':'joined_split_letters','source_start':prev['end'],
                                   'source_end':w['start'],'page':line.page})
                elif j not in joins and not smallcap and not inline_smallcap:emit(' ')
                else:events.append({'kind':'joined_small_caps','source_start':prev['end'],
                                    'source_end':w['start'],'page':line.page})
            raw=w['value'];value=values[w['index']]
            emit(value,w,'copy' if value==raw else 'normalized');seen.append(w['index'])
            if value!=raw:
                for kind,changed in [('expanded_ligature',any(c in LIGATURES for c in raw)),
                                     ('mapped_symbol_piece',any(ord(c) in SYMBOL_PIECES for c in raw)),
                                     ('composed_negation',NEGATION in raw or ops[w['index']].get('negate_first')),
                                     ('dropped_unmapped_glyph','�' in raw),
                                     ('dropped_invisible_character',INVISIBLE.search(raw) is not None)]:
                    if changed:events.append({'kind':kind,'source_start':w['start'],'page':line.page})
            if '�' in raw or any(0xe000<=ord(c)<=0xf8ff for c in value):
                flags.append({'kind':'damaged_glyph','source_start':w['start'],'source_end':w['end'],'page':line.page})
        regions.append({'start':start,'end':offset,'page':line.page,'source_line':line.original,
                        'layout_region':layout_regions.get(line.original),
                        'kind':'prose_candidate' if _prose(line) else 'preserved_line'})
        prior=line
    for kind,pattern in [('broken_math_font_encoding',BROKEN_MATH_FONT),('latex_source_text',LATEX_SOURCE)]:
        count=len(pattern.findall(text))
        if count>=3:flags.append({'kind':kind,'count':count})
    clean_text=''.join(chunks)
    deleted=[{'word_index':i,'source_start':artifact['rectangles'][i]['start'],
              'source_end':artifact['rectangles'][i]['end'],'reason':reason} for i,reason in sorted(removed.items())]
    if len(seen)!=len(set(seen)) or set(seen)|set(removed)!=set(range(len(artifact['rectangles']))):
        raise AssertionError('Lost or duplicated source words')
    result={'format':VERSION+('+mineru-figures' if figure_regions else ''),'text':clean_text,'text_sha256':hashlib.sha256(clean_text.encode()).hexdigest(),
            'source_text_sha256':source_hash,'pdf_sha256':artifact['pdf_sha256'],
            'offset_unit':'unicode_code_points','mapping':mapping,'regions':regions,
            'removed':deleted,'events':events,'flags':flags,
            'stats':{'source_words':len(artifact['rectangles']),'retained_words':len(seen),
                     'removed_words':len(removed),'rules':dict(Counter(removed.values())+Counter(e['kind'] for e in events))}}
    validate(result,artifact)
    return result


def validate(result,artifact):
    """Independent per-token content and coverage verification, run for every output."""
    text=artifact['text'];clean_text=result['text'];seen=set();end=0
    dehyphens={(e['source_start'],e['source_end']) for e in result['events'] if e['kind']=='dehyphenated'}
    for m in result['mapping']:
        if not end<=m['start']<m['end']<=len(clean_text):raise AssertionError('Bad output mapping')
        if clean_text[end:m['start']].strip():raise AssertionError('Unmapped output content')
        i=m['word_index'];w=artifact['rectangles'][i]
        if i in seen:raise AssertionError('Duplicate word')
        seen.add(i)
        if m['source_start']!=w['start'] or m['source_end'] not in [w['end'],w['end']-1]:raise AssertionError('Source mapping changed')
        if m['source_end']!=w['end'] and (m['source_end'],w['end']) not in dehyphens:raise AssertionError('Unexplained character deletion')
        expected=_normalize(text[m['source_start']:m['source_end']],
                            m.get('strip_negation',False),m.get('negate_first',False))
        if clean_text[m['start']:m['end']]!=expected:raise AssertionError('Unexplained text modification')
        end=m['end']
    removed={r['word_index'] for r in result['removed']}
    if seen & removed or seen|removed!=set(range(len(artifact['rectangles']))):raise AssertionError('Source coverage mismatch')
    if clean_text[end:].strip():raise AssertionError('Unmapped output tail')
    return True


def source_spans(result,start,end):
    """Project a clean range to original spans, retaining noncontiguous order.

    A partially selected ligature-expanded token maps to its whole source token.
    Synthetic whitespace has no source span. Consumers must use this function,
    not canonical word offsets, when highlighting derived text.
    """
    if not 0<=start<=end<=len(result['text']):raise ValueError('Invalid clean range')
    spans=[]
    for m in result['mapping']:
        a=max(start,m['start']);b=min(end,m['end'])
        if a>=b:continue
        if m['kind']=='copy':x=m['source_start']+a-m['start'];y=x+b-a
        else:x=m['source_start'];y=m['source_end']
        spans.append({'start':x,'end':y,'page':m['page']})
    return spans
