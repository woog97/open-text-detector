"""Positioned text artifacts. Offsets are Unicode code points, rectangles are PDF points."""
import csv
import difflib
import hashlib
import io
import os
import re
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from .result_codec import encode, decode

FORMAT = 'positioned-text-v1'

def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def read_text(path):
    path = Path(path)
    return decode(path.read_bytes())['text'] if path.suffix == '.pgf' else path.read_text()

def _tokens(text):
    return [(unicodedata.normalize('NFKC', m.group()), m.start(), m.end()) for m in re.finditer(r'\S+', text)]

def map_legacy(canonical, legacy):
    """Only exact normalized token blocks unique on both corresponding pages are mapped.

    No fuzzy spelling, cross-page matching, or geometry guessed for unmatched text.
    """
    output = {k: v for k, v in canonical.items() if k not in ('text', 'text_sha256', 'rectangles', 'mapping')}
    output.update(text=legacy, text_sha256=text_hash(legacy), rectangles=[])
    pieces = list(re.finditer(r'\n\f', legacy))
    ranges = []; start = 0
    for sep in pieces:
        ranges.append((start, sep.start())); start = sep.end()
    if legacy[start:].strip(): ranges.append((start, len(legacy)))
    if len(ranges) != len(canonical['pages']):
        output['mapping'] = {'kind':'legacy-exact-blocks', 'status':'unmapped', 'reason':'page_count_mismatch', 'mapped_nonspace_chars':0}
        return output
    bypage = {}
    for r in canonical['rectangles']: bypage.setdefault(r['page'], []).append(r)
    mapped = 0
    for page, (a,b) in zip(canonical['pages'], ranges):
        source = _tokens(legacy[a:b]); rects = bypage.get(page['page'], [])
        keys = [unicodedata.normalize('NFKC', canonical['text'][r['start']:r['end']]) for r in rects]
        wanted = [t[0] for t in source]
        # Count full block occurrences, including overlaps. Only one occurrence is safe.
        def unique(hay, needle):
            n=len(needle); count=0
            for i in range(len(hay)-n+1):
                if hay[i:i+n]==needle:
                    count+=1
                    if count>1:return False
            return count==1
        for block in difflib.SequenceMatcher(None, wanted, keys, autojunk=False).get_matching_blocks():
            if not block.size:continue
            seq=wanted[block.a:block.a+block.size]
            if not unique(wanted,seq) or not unique(keys,seq):continue
            # Isolated repeated/common short words provide insufficient evidence.
            if block.size<3 and sum(map(len,seq))<16:continue
            for j in range(block.size):
                _,x,y=source[block.a+j];r=rects[block.b+j]
                output['rectangles'].append({**r,'start':a+x,'end':a+y});mapped+=y-x
    total=sum(not c.isspace() for c in legacy)
    output['mapping']={'kind':'legacy-exact-blocks','status':'complete' if mapped==total else 'partial','mapped_nonspace_chars':mapped,'total_nonspace_chars':total,'coverage':mapped/max(1,total),'notice':'Unmapped characters must not be highlighted. Exact normalized blocks only; no fuzzy matching.'}
    return output

def page_geometry(pdf, pages):
    from pypdf import PdfReader
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    original=PdfReader(str(pdf)).pages
    assert len(original)==len(pages)
    result=[]
    for metadata, source in zip(pages,original):
        rotation=int(source.rotation)%360
        width=metadata['width'];height=metadata['height']
        if rotation in (90,270):width,height=height,width
        result.append({**metadata,'width':width,'height':height,'rotation':rotation,
                       'media_box':[float(x) for x in source.mediabox],
                       'crop_box':[float(x) for x in source.cropbox]})
    return result

def extract(pdf, ocr_pages=None, timeout=180):
    pdf=Path(pdf);digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='positioned-pdf-') as tmp:
        bbox=Path(tmp)/'words.xhtml'
        subprocess.run(['pdftotext','-enc','UTF-8','-bbox-layout',str(pdf),str(bbox)],check=True,capture_output=True,timeout=timeout)
        raw=bbox.read_text()
        # Preserve unsupported font codes as visible replacement characters, never delete them.
        raw=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '\ufffd', raw)
        root=ET.fromstring(raw)
        pages=[];chunks=[];rects=[];offset=0
        elements=[p for p in root.iter() if p.tag.rsplit('}',1)[-1]=='page']
        method='poppler-bbox-layout'
        for number,page in enumerate(elements,1):
            width=float(page.attrib['width']);height=float(page.attrib['height'])
            pages.append({'page':number,'width':width,'height':height})
            lines=[x for x in page.iter() if x.tag.rsplit('}',1)[-1]=='line']
            for line in lines:
                words=[w for w in line if w.tag.rsplit('}',1)[-1]=='word']
                for w in words:
                    text=''.join(w.itertext())
                    if not text.strip():continue
                    rects.append({'start':offset,'end':offset+len(text),'page':number,**{k:float(w.attrib[v]) for k,v in [('x0','xMin'),('y0','yMin'),('x1','xMax'),('y1','yMax')]}})
                    chunks.append(text+' ');offset+=len(text)+1
                chunks.append('\n');offset+=1
            chunks.append('\f');offset+=1
        pages=page_geometry(pdf,pages)
        selected_ocr=set(ocr_pages or [])
        if not rects: selected_ocr={p['page'] for p in pages}
        if selected_ocr:
            method='tesseract-english-300dpi-tsv' if len(selected_ocr)==len(pages) else 'poppler-bbox-layout-with-page-ocr'
            native_text=''.join(chunks);native_rects=rects
            chunks=[];rects=[];offset=0
            for p in pages:
                if p['page'] not in selected_ocr:
                    previous=None
                    for box in (r for r in native_rects if r['page']==p['page']):
                        if previous is not None:
                            gap=native_text[previous:box['start']]
                            chunks.append(gap);offset+=len(gap)
                        word=native_text[box['start']:box['end']]
                        rects.append({**box,'start':offset,'end':offset+len(word)})
                        chunks.append(word);offset+=len(word);previous=box['end']
                    chunks.append('\n\f');offset+=2
                    continue
                image=Path(tmp)/'page'
                subprocess.run(['pdftoppm','-f',str(p['page']),'-l',str(p['page']),'-singlefile','-r','300','-png',str(pdf),str(image)],check=True,capture_output=True,timeout=120)
                from PIL import Image
                with Image.open(str(image)+'.png') as im:
                    sx=p['width']/im.width;sy=p['height']/im.height
                    # Light shaded figure backgrounds otherwise cause Tesseract to
                    # discard embedded text as an image region.
                    cleaned=im.convert('L').point(lambda value:255 if value>190 else 0)
                cleaned.save(str(image)+'.png')
                r=subprocess.run(['tesseract',str(image)+'.png','stdout','-l','eng','--psm','3','tsv'],check=True,capture_output=True,text=True,timeout=120)
                previous=None
                for w in csv.DictReader(io.StringIO(r.stdout),delimiter='\t',quoting=csv.QUOTE_NONE):
                    text=w.get('text','')
                    if w['level']!='5' or not text.strip():continue
                    line=(w['block_num'],w['par_num'],w['line_num'])
                    if previous is not None and line!=previous:chunks.append('\n');offset+=1
                    previous=line;x=float(w['left'])*sx;y=float(w['top'])*sy
                    rects.append({'start':offset,'end':offset+len(text),'page':p['page'],'x0':x,'y0':y,'x1':x+float(w['width'])*sx,'y1':y+float(w['height'])*sy})
                    chunks.append(text+' ');offset+=len(text)+1
                chunks.append('\n\f');offset+=2
        for box in rects:
            box['x0'],box['x1']=sorted([box['x0'],box['x1']])
            box['y0'],box['y1']=sorted([box['y0'],box['y1']])
        text=''.join(chunks)
        assert rects,'No positioned text extracted'
        return {'format':FORMAT,'geometry_version':2,'pdf_sha256':digest,'text':text,'text_sha256':text_hash(text),'offset_unit':'unicode_code_points','coordinates':{'unit':'pdf_points','origin':'top_left','page_numbering':'one_based','page_box':'media_box','rotation':'as_rendered_by_poppler'},'pages':pages,'rectangles':rects,'method':method,'ocr_pages':sorted(selected_ocr),'ocr_preprocessing':'grayscale-threshold-190' if selected_ocr else None,'mapping':{'kind':'canonical','status':'complete','notice':'Word rectangles; only whitespace has no rectangle. Image text inside otherwise text-bearing native pages is not OCRed.'}}

def save(artifact, folder):
    folder=Path(folder)/artifact['pdf_sha256'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/(artifact['text_sha256']+'.pgf');blob=encode(artifact, level=9)
    if decode(blob)!=artifact:raise ValueError('Extraction roundtrip failed')
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        stream.write(blob);stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    if decode(path.read_bytes())!=artifact:raise ValueError('Persisted extraction failed verification')
    return path

def find_maps(text, dataset_dir):
    """Content-addressed links; results need not duplicate geometry for each model."""
    if not text.strip():return []
    from .sqlite_runtime import sqlite3
    database=Path(dataset_dir).parent/'extractions/positioned/index.sqlite3'
    if not database.is_file():return []
    with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as conn:
        rows=conn.execute('SELECT pdf_sha256,text_sha256,blob_sha256,bytes,mapping_status,coverage FROM artifacts WHERE text_sha256=?',(text_hash(text),)).fetchall()
    return [{'format':FORMAT,'pdf_sha256':r[0],'text_sha256':r[1],'sha256':r[2],'bytes':r[3],'mapping_status':r[4],'coverage':r[5],'url':f'/v1/extractions/{r[0]}/{r[1]}','offset_unit':'unicode_code_points'} for r in rows]
