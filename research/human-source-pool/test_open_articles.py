import unittest
from ingest_open_articles import eligible,pairs,normalized_title

class ACLTests(unittest.TestCase):
    def row(self):
        return dict(acl_id='P19-1001',publisher='Association for Computational Linguistics',year='2019',title='A Useful Paper',language='English',full_text=('The researchers discuss the approach and the results of the study. '*150)+'\nReferences\nNEVER INCLUDE REFERENCE LIST',author='Author',doi='10.18653/v1/P19-1001')
    def test_acl_year_publisher_and_proceedings(self):
        e={'ids':set(),'titles':set()};r=self.row();self.assertIsNone(eligible(r,e))
        for changes in [dict(year='2022'),dict(publisher='ELRA'),dict(acl_id='W19-1001'),dict(acl_id='2020.lrec-1.100')]:
            self.assertIsNotNone(eligible({**r,**changes},e))
    def test_protected_id_and_normalized_title(self):
        r=self.row()
        self.assertEqual(eligible(r,{'ids':{'P19-1001'},'titles':set()}),'protected_family')
        self.assertEqual(eligible(r,{'ids':set(),'titles':{normalized_title('A {Useful} Paper')}}),'protected_family')
    def test_unchanged_offsets_quarantine_and_references(self):
        r=self.row();m=dict(source_file='source.parquet',retrieved_at='2026-10-02',license_evidence='CC BY4 with mirror NC restriction')
        results=list(pairs(r,0,m,[10,10,10,10]));self.assertTrue(results);self.assertLessEqual(len(results),3)
        for out,raw in results:
            self.assertEqual(r['full_text'][out['raw_start']:out['raw_end']],out['text'])
            self.assertNotIn('NEVER INCLUDE',out['text']);self.assertEqual(out['source_id'],'acl');self.assertEqual(out['category'],'scientific');self.assertFalse(out['training_eligible'])
            self.assertEqual(raw['record']['text'],r['full_text'])

if __name__=='__main__':unittest.main()
