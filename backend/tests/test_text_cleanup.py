"""Regression fixtures are synthetic; no third-party paper text is committed."""
import copy
import re
import hashlib
import pytest
from open_text_detector_backend import text_cleanup
from open_text_detector_backend.text_cleanup import clean, source_spans, validate


def artifact(lines, pages=2):
    """lines: (page, y, [(text,x,width,height), ...]); source order is intentional."""
    text='';rectangles=[]
    for page,y,words in lines:
        for token,x,width,height in words:
            rectangles.append(dict(start=len(text),end=len(text)+len(token),page=page,
                                   x0=x,y0=y,x1=x+width,y1=y+height))
            text+=token+' '
        text+='\n'
    return dict(text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                pdf_sha256='a'*64,rectangles=rectangles,
                pages=[dict(page=i,width=612.,height=792.) for i in range(1,pages+1)])


def line(s,page=1,y=100,x=108,height=10):
    words=[]
    for t in s.split():
        width=len(t)*height*.45;words.append((t,x,width,height));x+=width+height*.3
    return page,y,words


def test_remove_only_confirmed_margin_rails_and_headers():
    ls=[]
    for p in [1,2]:
        ls.append(line('Under review as a conference paper at ICLR 2027',p,28))
        ls.extend(line(f'{i+(p-1)*54:03}',p,81+i*12,73,height=9) for i in range(54))
        ls.append(line('1 123 2027 0.91 scientific results remain',p,150))
        ls.append(line(str(p),p,755,304,height=10))
    a=artifact(ls);before=copy.deepcopy(a);r=clean(a)
    assert a==before
    assert r['stats']['rules']['review_line_number']==108
    assert r['stats']['rules']['page_number']==2
    assert 'scientific results remain' in r['text'] and '123 2027 0.91' in r['text']
    assert 'Under review' not in r['text']
    assert validate(r,a)


def test_preserve_numbers_in_tables_short_margin_lists_and_irregular_sequences():
    ls=[line(str(i),y=100+i*12,x=200) for i in range(100,114)]
    ls += [line(f'{i:03}',y=100+i*12,x=73) for i in range(5)]
    ls += [line(f'{100+i*2:03}',y=200+i*12,x=73) for i in range(20)]
    a=artifact(ls);r=clean(a)
    assert r['removed']==[]
    assert r['stats']['retained_words']==len(a['rectangles'])


def test_header_body_quotes_and_single_page_headers_preserved():
    s='Under review as a conference paper at ICLR 2027'
    a=artifact([line(s,y=200),line(s,page=2,y=200),line(s,y=28)])
    assert clean(a)['text'].count(s)==3


def test_only_bottom_centre_physical_page_numbers_removed():
    a=artifact([line('1',1,300,304),line('2',2,300,304),line('13',1,755,304),line('14',2,755,304)])
    assert not clean(a)['removed']


def test_heading_ligature_math_and_legitimate_capitals():
    a=artifact([line('A BSTRACT'),line('I NTRODUCTION',y=120),line('A B testing C D',y=140),
                line('ﬁnal ﬂow ² 𝑥 − � \uf8f4',y=160),line('REFERENCES',y=180)])
    r=clean(a)
    assert 'ABSTRACT' in r['text'] and 'INTRODUCTION' in r['text']
    assert 'A B testing C D' in r['text']
    assert 'final flow ² 𝑥 − ⎪' in r['text']  # Symbol-font brace piece mapped
    assert r['stats']['rules']['unmapped_glyph']==1 and not r['flags']
    start=r['text'].index('final');sp=source_spans(r,start,start+2)
    assert a['text'][sp[0]['start']:sp[0]['end']]=='ﬁnal'


def test_tight_small_caps_only():
    a=artifact([(1,80,[('W',108,12,16),('ORLD',120.1,30,12),('ABC',158,25,16),('DE',190,15,16)])])
    a['rectangles'][1].update(y0=83.,y1=95.)
    assert clean(a)['text']=='WORLD ABC DE'


def test_dehyphen_requires_evidence_and_preserves_compounds():
    a=artifact([line('These robust experiments demonstrate a clear improve-',y=100),
                line('ment across several carefully selected experimental conditions',y=112),
                line('improvement improvement high-resource high-resource',y=160),
                line('These robust experiments demonstrate a clear high-',y=200),
                line('resource method using several carefully selected experimental conditions',y=212)])
    r=clean(a)
    assert 'improvement across' in r['text']
    assert 'high-resource method' in r['text']  # preserve the attested compound's hyphen
    assert r['stats']['rules']['dehyphenated']==1
    i=r['text'].index('improvement across');sp=source_spans(r,i,i+11)
    assert ''.join(a['text'][s['start']:s['end']] for s in sp)=='improvement'


def test_unknown_hyphen_and_paragraph_indent_unchanged():
    a=artifact([line('These robust experiments demonstrate a clear unobtain-',y=100),
                line('able result across several carefully selected experimental conditions',y=112),
                line('This is a long paragraph that should not merge into',y=150),
                line('An indented paragraph begins a different thought entirely',y=162,x=125)])
    r=clean(a)
    assert 'unobtain-\nable' in r['text']
    assert 'into\nAn indented' in r['text']


def test_separate_interleaved_figure_from_strong_prose_lane():
    ls=[]
    for i in range(7):
        ls.append(line(f'This prose line describes reliable method number {i}',y=100+i*12,x=108,height=10))
        ls.append(line(f'Diagram label {i}',y=102+i*12,x=400,height=5))
    a=artifact(ls);r=clean(a)
    assert r['stats']['rules']['separated_prose_lane']==1
    assert r['text'].index('number 6')<r['text'].index('Diagram label 0')
    assert r['stats']['retained_words']==len(a['rectangles'])
    assert validate(r,a)


def test_ambiguous_overlapping_region_does_not_reorder():
    ls=[]
    for i in range(7):
        ls.append(line(f'This prose line describes reliable method number {i}',y=100+i*12))
        ls.append(line(f'Equation {i}',y=102+i*12,x=200,height=5))
    r=clean(artifact(ls))
    assert not r['stats']['rules'].get('separated_prose_lane')
    assert any(f['kind']=='ambiguous_reading_order' for f in r['flags'])
    assert r['text'].index('Equation 0')<r['text'].index('number 1')


def test_complete_columns_do_not_interleave():
    ls=[line(f'Left column contains independent scientific discussion number {i}',y=100+i*12,x=50,height=8) for i in range(7)]
    ls += [line(f'Right column contains independent scientific discussion number {i}',y=100+i*12,x=330,height=8) for i in range(7)]
    r=clean(artifact(ls))
    assert not r['stats']['rules'].get('separated_prose_lane')
    assert r['text'].index('number 6')<r['text'].index('Right column')


def test_restore_only_unambiguous_top_title():
    a=artifact([line('A BSTRACT',y=200),line('Several scientific sentences describe the new method',y=230),
                line('A RELIABLE SCIENTIFIC METHOD',y=80,height=16)])
    r=clean(a);assert r['text'].startswith('A RELIABLE SCIENTIFIC METHOD')
    a=artifact([line('Existing text is already above the heading',y=60),line('A LATER SECTION TITLE',y=80,height=16)])
    assert clean(a)['text'].startswith('Existing text')


def test_corrupt_hash_or_unmapped_source_rejected():
    a=artifact([line('Correct source text')]);a['text_sha256']='0'*64
    with pytest.raises(ValueError):clean(a)
    a=artifact([line('Correct source text')]);a['rectangles']=a['rectangles'][1:]
    with pytest.raises(ValueError):clean(a)


def test_roundtrip_validation_catches_changed_or_lost_words():
    a=artifact([line('Correct source text')]);r=clean(a);r['text']=r['text'].replace('Correct','Corrupt')
    with pytest.raises(AssertionError):validate(r,a)


def test_deterministic_and_original_offsets_project_exactly():
    a=artifact([line('Correct source text')]);assert clean(a)==clean(a)
    r=clean(a);i=r['text'].index('source');sp=source_spans(r,i+1,i+4)
    assert len(sp)==1 and a['text'][sp[0]['start']:sp[0]['end']]=='our'
    assert source_spans(r,0,0)==[]


def test_sequential_margin_table_labels_are_not_review_numbers():
    # Same row labels repeated in a table on each page, not globally increasing.
    a=artifact([line(f'{i:03}',page=p,y=90+i*12,x=73,height=9) for p in [1,2] for i in range(30)])
    assert not clean(a)['removed']


def test_math_superscripts_not_joined_like_small_caps():
    a=artifact([(1,80,[('LAMBDA',108,50,16),('BETA',158.1,25,12)])])
    a['rectangles'][1].update(y0=75,y1=87)
    assert clean(a)['text']=='LAMBDA BETA'


def test_side_region_boundaries_remain_explicit():
    ls=[]
    for i in range(7):
        ls.extend([line(f'This prose line describes reliable method number {i}',y=100+i*12),
                   line(f'Diagram label {i}',y=102+i*12,x=400,height=5)])
    r=clean(artifact(ls))
    assert '\n\nDiagram label 0' in r['text']
    assert any((x['layout_region'] or '').startswith('side-region-') for x in r['regions'])


def test_correct_table_row_with_wide_gaps_is_preserved():
    a=artifact([(1,100,[('Method',108,30,10),('Accuracy',300,40,10),('0.95',450,25,10)])])
    assert clean(a)['text']=='Method Accuracy 0.95'


def test_inline_small_math_does_not_split_a_body_line():
    a=artifact([(1,100,[('These',108,25,10),('values',138,25,10),('are',168,15,10),
                       ('defined',188,30,10),('x',240,5,5),('y',250,5,5)])])
    assert clean(a)['text']=='These values are defined x y'


def test_invisible_characters_and_negation_slash_repaired():
    a=artifact([line('Vector\ufe01 y\u02c6\ufe01 zero\u200bwidth i\u0338 = j and ) \u0338 = k \u0338\u2208 S',y=100)])
    r=clean(a)
    assert r['text']=='Vector y\u02c6 zerowidth i \u2260 j and ) \u2260 k \u2209 S'
    assert r['stats']['rules']['combined_negation']==1
    assert validate(r,a)
    i=r['text'].index('\u2260');sp=source_spans(r,i,i+1)
    assert a['text'][sp[0]['start']:sp[0]['end']]=='='


def test_venue_running_headers_removed():
    for header in ['Published in Transactions on Machine Learning Research (06/2024)',
                   'Published as a conference paper at COLM 2025']:
        a=artifact([line(header,p,28) for p in [1,2]]+[line('Body text stays',p,300) for p in [1,2]])
        r=clean(a)
        assert header not in r['text'] and r['text'].count('Body text stays')==2


def test_checklist_template_lines_removed_answers_kept(monkeypatch):
    template=[f'template guidance sentence number {i} for every author' for i in range(21)]
    shingles=frozenset(h for t in template for h in text_cleanup._shingles(t))
    monkeypatch.setattr(text_cleanup,'_checklist_template',lambda:(shingles,frozenset([text_cleanup._hash('Guidelines:')])))
    ls=[line('Guidelines:',y=80)]+[line(t,y=100+12*i) for i,t in enumerate(template)]
    ls+=[line('Answer: [Yes]',y=300),line('Justification: our own explanation of the method',y=312)]
    r=clean(artifact(ls))
    assert r['text']=='Answer: [Yes]\nJustification: our own explanation of the method'
    # Too few template lines: nothing removed.
    assert 'template guidance' in clean(artifact(ls[:5]+ls[-2:]))['text']


def test_inline_small_caps_joined_but_not_subscripts():
    a=artifact([(1,100,[('We',100,13,12),('propose',116,31.5,12),('D',150,7.2,12),('IST',157.7,12.9,9.6),('here',173,20,12)])])
    a['rectangles'][3].update(y0=102.4,y1=112.)
    assert clean(a)['text']=='We propose DIST here'
    a=artifact([(1,100,[('and',100,14,9.4),('L',117,6.9,9.4),('CAR',123.9,18.2,6.2),('denotes',145,30,9.4)])])
    a['rectangles'][2].update(y0=103.2+.9,y1=110.3)
    assert clean(a)['text']=='and L CAR denotes'
    # Overlapping capital, single small cap and dash-joined compound in a heading.
    a=artifact([(1,100,[('A',373.96,12.68,15.39),('TTRIBUTION',385.81,91.94,12.31),('I',490,4,15.39),('S',494.6,5.3,12.31),
                        ('G',108.43,12.43,15.39),('ROMOV',121.73,53.42,12.31),('-W',174.44,22.85,15.39),('ASSERSTEIN',196.3,89.5,12.31)])])
    for k in (0,2,4,6):a['rectangles'][k].update(y0=81.88,y1=97.27)
    for k in (1,3,5,7):a['rectangles'][k].update(y0=84.21,y1=96.53)
    assert clean(a)['text']=='ATTRIBUTION IS GROMOV-WASSERSTEIN'


def test_split_letters_joined_horizontally_and_vertically():
    xs=[('w',202.4,4.8),('i',208.1,1.5),('t',210.4,1.9),('h',213.1,3.7),('ou',217.6,7.8),('t',226.1,1.9),
        ('du',230.5,8.1),('pl',239.3,5.8),('i',245.9,1.5),('cat',248.3,9.3),('i',258.4,1.5),('n',260.8,3.7),('g',265.2,3.7)]
    a=artifact([(1,238,[(t,x,w,6.2) for t,x,w in xs])])
    assert clean(a)['text']=='without duplicating'
    ys=[('F',145.7),('i',143.2),('x',140.7),('R',137.0),('e',134.5),('s',132.0)]  # rotated label, bottom-up
    a=artifact([(1,0,[(t,354,6.1,2.3) for t,y in ys])])
    for box,(t,y) in zip(a['rectangles'],ys):box.update(y0=y,y1=y+2.3)
    assert clean(a)['text']=='Fix Res'
    tracked=[(c,137.99+5.97*i,4.78,6.15) for i,c in enumerate('TEMPLATES/SUPERVISOR')]
    assert clean(artifact([(1,546,tracked)]))['text']=='TEMPLATES/SUPERVISOR'
    # Rotated text stored top-down (reversed reading order) is left alone.
    a=artifact([(1,0,[(t,354,6.1,2.3) for t,y in ys])])
    for box,(t,y) in zip(a['rectangles'],reversed(ys)):box.update(y0=y,y1=y+2.3)
    assert clean(a)['text']=='F i x R e s'
    # Ordinary short words with normal spacing are untouched.
    assert clean(artifact([line('a to in of an it is as')]))['text']=='a to in of an it is as'


def test_document_flags_for_broken_math_fonts():
    r=clean(artifact([line('s t\u00b41 and a t\u201c1 then q\u00b41 or x \u00b42')]))
    assert any(f['kind']=='broken_math_font_encoding' for f in r['flags'])


def rail(page,numbers,fused=()):
    """A review margin rail; numbers listed in `fused` share a source line with body text."""
    out=[]
    for k,n in enumerate(numbers):
        y=81+k*12;words=[(f'{n:03}',73,13.3 if n<1000 else 17.8,7.5)]
        if n in fused:words.append(('Figure',300,30,7.5))
        out.append((page,y,words))
    return out


def test_review_numbers_fused_four_digit_and_restarted_rails_removed():
    ls=rail(1,range(0,54),fused={10,11})+rail(2,range(54,108))
    ls+=rail(3,range(0,54))+rail(4,range(54,108))       # appended supplement restarts at 000
    ls+=rail(19,range(972,1026))+rail(20,range(1026,1080))  # 3- to 4-digit transition
    a=artifact(ls,pages=20);r=clean(a)
    assert r['stats']['rules']['review_line_number']==6*54
    assert r['text'].count('Figure')==2 and not re.search(r'\d',r['text'])
    assert validate(r,a)


def test_offset_numbering_continues_across_pages():
    # Numbering lags the page index after unnumbered pages, but stays continuous.
    a=artifact(rail(30,range(1482,1536))+rail(31,range(1536,1590)),pages=31)
    assert clean(a)['stats']['rules']['review_line_number']==108


def test_anonymous_author_placeholder_removed_only_as_top_pair():
    a=artifact([line('A RELIABLE METHOD',y=80,height=16),line('Anonymous authors',y=120),
                line('Paper under double-blind review',y=132),line('Anonymous authors',y=600)])
    assert clean(a)['text']=='A RELIABLE METHOD\nAnonymous authors'


def test_figure_regions_remove_figure_text_and_keep_caption_and_prose():
    from open_text_detector_backend.text_cleanup import mineru_figure_regions
    a=artifact([line('Body prose describing the experiment in detail here',y=100,x=108),
                line('0.2 0.4 0.6 Accuracy',y=300,x=250),            # inside the chart body
                line('Figure 2: Accuracy over training steps',y=420,x=150)])
    middle={'pages':[{'page_idx':0,'blocks':[{'type':'chart','bbox':[.3,.33,.8,.6],'content':[
        {'type':'chart_body','bbox':[.35,.35,.75,.42]},{'type':'chart_caption','bbox':[.2,.52,.8,.56]}]}]}]}
    regions=mineru_figure_regions(middle)
    assert regions=={1:[[.35,.35,.75,.42]]}
    r=clean(a,figure_regions=regions)
    assert 'Accuracy over training steps' in r['text'] and 'Body prose' in r['text']
    assert '0.4' not in r['text'] and r['stats']['rules']['figure_region']==4
    assert r['format'].endswith('+mineru-figures') and validate(r,a)
    assert clean(a)['text'].count('Accuracy')==2   # unchanged without regions
