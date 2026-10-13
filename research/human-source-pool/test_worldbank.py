import tempfile
import unittest
from pathlib import Path
from ingest_worldbank import eligible_doc,text_url,license_grant,pairs
from expand_pool import connect,add


def document():
    return {'id':'123','lang':'English','docdt':'2018-01-01T00:00:00Z','disclstat':'Disclosed',
            'display_title':'Development report','authors':{'0':{'author':'Example Author'}},
            'dois':'10.1596/example','txturl':'http://documents.worldbank.org/curated/en/123456/text/report.txt'}


def sample_text():
    grant='This work is available under the Creative Commons Attribution 3.0 IGO license (CC BY 3.0 IGO), http://creativecommons.org/licenses/by/3.0/igo/.'
    paras=['The report explains how the public institutions and people can work together to improve the economy and its services. '*8+str(i) for i in range(24)]
    return grant+'\r\n\r\n'+'\r\n\r\n'.join(paras)


class WorldBankTests(unittest.TestCase):
    def test_per_work_grant_not_cited_license(self):
        self.assertIsNotNone(license_grant(sample_text()))
        self.assertIsNotNone(license_grant(sample_text().replace('http://creativecommons','http://\r\ncreativecommons')))
        self.assertEqual(license_grant(sample_text().replace('/3.0/igo/','/3.0/\r\nigo/'))['url'],'https://creativecommons.org/licenses/by/3.0/igo/')
        self.assertIsNone(license_grant('References discuss http://creativecommons.org/licenses/by/3.0/igo/'))
        self.assertIsNone(license_grant(sample_text().replace('licenses/by/3.0/igo/','licenses/by-nc/4.0/')))

    def test_date_language_and_disclosure(self):
        self.assertTrue(eligible_doc(document()))
        for key,value in [('docdt','2024-01-01'),('lang','Spanish'),('disclstat','Restricted')]:
            row=document();row[key]=value;self.assertFalse(eligible_doc(row))

    def test_official_url_only(self):
        self.assertEqual(text_url(document()),'https://documents1.worldbank.org/curated/en/123456/text/report.txt')
        row=document();row['txturl']='https://other.example/report.txt'
        with self.assertRaises(ValueError):text_url(row)

    def test_CRLF_exact_offsets_nonoverlap_and_provenance(self):
        text=sample_text();manifest={'sha256':'original-bytes-hash','retrieved_at':'2026-10-02','url':text_url(document())}
        results=list(pairs(document(),text,manifest,[5,5,5,5]));self.assertTrue(results);self.assertLessEqual(len(results),3)
        ranges=[]
        for row,raw in results:
            a,b=row['raw_start'],row['raw_end'];self.assertEqual(text[a:b],row['text']);self.assertEqual(row['word_count'],len(row['text'].split()))
            self.assertFalse(any(a<e and b>s for s,e in ranges));ranges.append((a,b))
            self.assertEqual(raw['record']['text'],text);self.assertIn('license_grant',raw['record']['metadata'])
            self.assertEqual(row['source_id'],'worldbank');self.assertEqual(row['category'],'professional');self.assertFalse(row['training_eligible'])
            self.assertNotIn('Creative Commons',row['text'])

    def test_dedup_and_source_cap(self):
        pair=next(pairs(document(),sample_text(),{'sha256':'hash','retrieved_at':'now','url':text_url(document())},[5]*4))
        with tempfile.TemporaryDirectory() as tmp:
            db=connect(Path(tmp)/'db')
            with db:self.assertTrue(add(db,*pair,1));self.assertFalse(add(db,*pair,1))
            db.close()

    def test_long_passage_retains_original_page_breaks(self):
        text=sample_text().replace('\r\n\r\n','\r\n\r\n12 Running title\f\r\n\r\n')
        result=list(pairs(document(),text,{'sha256':'hash','retrieved_at':'now','url':text_url(document())},[0,0,0,3]))
        self.assertTrue(result)
        for row,raw in result:
            self.assertEqual(text[row['raw_start']:row['raw_end']],row['text'])
            self.assertIn('\f',row['text']);self.assertIn('publisher_page_break_inside_contiguous_span',row['reason_codes'])


if __name__=='__main__':unittest.main()
