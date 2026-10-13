import tempfile
import unittest
from pathlib import Path
from ingest_institutional import speech_date,speech_links,extract_fed,pairs
from expand_pool import connect,add

URL='https://www.federalreserve.gov/newsevents/speech/waller20211217a.htm'


def html(speaker='Governor Example Author'):
    para='The policy report explains how the economy and its institutions work together for the public. '*8
    return '<div id="article"><div class="heading col-md-8"><h3 class="title">A policy speech</h3><p class="speaker">'+speaker+'</p></div><div class="col-md-8">'+''.join('<p>'+para+str(i)+'</p>' for i in range(16))+'<blockquote><p>THIRD PARTY EXCLUDED</p></blockquote><table><tr><td>TABLE EXCLUDED</td></tr></table><p><a name="f1"></a>FOOTNOTE EXCLUDED Return to text</p></div></div>'


class InstitutionalTests(unittest.TestCase):
    def test_official_precutoff_dates_only(self):
        self.assertEqual(speech_date(URL),'2021-12-17')
        for url in [URL.replace('20211217','20250101'),URL.replace('www.federalreserve.gov','other.example'),URL.replace('20211217','20211399')]:
            self.assertIsNone(speech_date(url))

    def test_index_filters_external_and_postcutoff(self):
        source='<a href="/newsevents/speech/waller20211217a.htm">official</a><a href="https://elsewhere.org/speech.htm">other</a><a href="/newsevents/speech/waller20250101a.htm">later</a>'
        self.assertEqual(speech_links(source,URL),[URL])

    def test_board_author_and_extraction(self):
        value=extract_fed(html(),URL)
        self.assertEqual(value['author'],'Governor Example Author')
        self.assertNotIn('EXCLUDED',value['text']);self.assertNotIn('A policy speech',value['text'])
        with self.assertRaises(ValueError):extract_fed(html('Guest speaker'),URL)

    def test_inline_footnote_reference_keeps_speech_body(self):
        page=html().replace('The policy report','<a name="f1">1</a> The policy report',1)
        result=extract_fed(page,URL)
        self.assertGreater(len(result['text'].split()),500)
        self.assertNotIn('FOOTNOTE EXCLUDED',result['text'])

    def test_offsets_provenance_and_quarantine(self):
        result=list(pairs(html(),URL,[9]*4));self.assertTrue(result);self.assertLessEqual(len(result),3)
        occupied=[]
        for row,raw in result:
            a,b=row['raw_start'],row['raw_end'];self.assertEqual(raw['record']['text'][a:b],row['text'])
            self.assertEqual(row['category'],'professional');self.assertEqual(row['source_id'],'fed')
            self.assertEqual(raw['record']['metadata']['html'],html());self.assertFalse(row['training_eligible'])
            self.assertFalse(any(a<end and b>start for start,end in occupied));occupied.append((a,b))

    def test_staged_dedup_and_quota(self):
        pair=next(pairs(html(),URL,[9]*4))
        with tempfile.TemporaryDirectory() as tmp:
            db=connect(Path(tmp)/'stage.sqlite3')
            with db:self.assertTrue(add(db,*pair,1));self.assertFalse(add(db,*pair,1))
            db.close()


if __name__=='__main__':unittest.main()
