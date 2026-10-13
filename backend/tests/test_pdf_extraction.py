from reportlab.pdfgen import canvas
from open_text_detector_backend.pdf_extraction import extract,map_legacy,save,read_text,text_hash
from open_text_detector_backend.result_codec import decode,encode

def test_real_pdf_geometry_roundtrip(tmp_path):
    pdf=tmp_path/'sample.pdf';c=canvas.Canvas(str(pdf),pagesize=(300,400))
    c.drawString(30,350,'Alpha beta gamma delta');c.showPage();c.drawString(40,300,'Second page words here');c.save()
    a=extract(pdf)
    assert len(a['pages'])==2
    first=a['rectangles'][0]
    assert a['text'][first['start']:first['end']]=='Alpha'
    assert first['page']==1 and first['x0']==30 and 30<first['y0']<60
    assert all(0<=r['start']<r['end']<=len(a['text']) for r in a['rectangles'])
    p=save(a,tmp_path/'objects');assert read_text(p)==a['text'];assert decode(p.read_bytes())==a
    legacy='Alpha beta gamma delta\n\fSecond page words here\n\f'
    b=map_legacy(a,legacy);assert b['text']==legacy and b['mapping']['coverage']==1
    assert b['text_sha256']==text_hash(legacy)
    assert decode(encode(b))==b

def test_unmatched_page_not_guessed(tmp_path):
    pdf=tmp_path/'sample.pdf';c=canvas.Canvas(str(pdf));c.drawString(30,500,'Alpha beta gamma');c.save()
    a=extract(pdf);b=map_legacy(a,'Other words on a different page\n\fExtra page\n\f')
    assert not b['rectangles'] and b['mapping']['status']=='unmapped'

def test_insertions_are_unmapped(tmp_path):
    pdf=tmp_path/'sample.pdf';c=canvas.Canvas(str(pdf));c.drawString(30,500,'Alpha beta gamma delta');c.save()
    a=extract(pdf);b=map_legacy(a,'Alpha beta gamma delta invented\n\f')
    assert b['mapping']['status']=='partial'
    assert all('invented' not in b['text'][r['start']:r['end']] for r in b['rectangles'])

def test_map_lookup_is_tied_to_exact_text_and_pdf(tmp_path):
    from open_text_detector_backend.pdf_extraction import find_maps
    from open_text_detector_backend.sqlite_runtime import sqlite3
    root=tmp_path/'extractions/positioned';root.mkdir(parents=True)
    c=sqlite3.connect(root/'index.sqlite3');c.execute('CREATE TABLE artifacts(pdf_sha256,text_sha256,blob_sha256,bytes,mapping_status,coverage)')
    h=text_hash('Exact text');c.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?)',('a'*64,h,'b'*64,100,'partial',.9));c.commit();c.close()
    maps=find_maps('Exact text',tmp_path/'data')
    assert len(maps)==1 and maps[0]['text_sha256']==h and maps[0]['mapping_status']=='partial'
    assert find_maps('Exact text ',tmp_path/'data')==[]

def test_api_requires_auth_and_rejects_traversal(workspace):
    client,headers,app,_=workspace
    assert client.get('/v1/extractions/'+'a'*64+'/'+'b'*64).status_code==401
    assert client.get('/v1/extractions/not-a-hash/not-a-hash',headers=headers).status_code==404

def test_rotated_page_dimensions_match_rendered_coordinates(tmp_path):
    pdf=tmp_path/'rotated.pdf';c=canvas.Canvas(str(pdf),pagesize=(300,400));c.setPageRotation(90)
    c.drawString(30,200,'Alpha beta gamma');c.save()
    a=extract(pdf);p=a['pages'][0]
    assert p['rotation']==90 and p['width']==300 and p['height']==400
    assert p['media_box']==[0.,0.,400.,300.]
    assert all(r['x1']<=300 and r['y1']<=400 for r in a['rectangles'])

def test_mirrored_text_has_ordered_rectangles(tmp_path):
    pdf=tmp_path/'mirrored.pdf';c=canvas.Canvas(str(pdf));c.translate(400,0);c.scale(-1,1)
    c.drawString(20,300,'Mirrored words here');c.save()
    a=extract(pdf)
    assert a['rectangles']
    assert all(r['x0']<=r['x1'] and r['y0']<=r['y1'] for r in a['rectangles'])

def test_selected_page_ocr_preserves_other_page_words_and_boxes(tmp_path):
    pdf=tmp_path/'mixed.pdf';c=canvas.Canvas(str(pdf),pagesize=(300,400))
    c.drawString(30,350,'Alpha beta gamma delta');c.showPage()
    c.drawString(40,300,'Second page readable words');c.save()
    native=extract(pdf);mixed=extract(pdf,ocr_pages=[2])
    assert mixed['ocr_pages']==[2]
    assert mixed['method']=='poppler-bbox-layout-with-page-ocr'
    first=lambda a:[(a['text'][r['start']:r['end']],r['x0'],r['y0'],r['x1'],r['y1']) for r in a['rectangles'] if r['page']==1]
    assert first(native)==first(mixed)
    assert 'Second page readable words' in mixed['text']
    assert all(0<=r['start']<r['end']<=len(mixed['text']) for r in mixed['rectangles'])
