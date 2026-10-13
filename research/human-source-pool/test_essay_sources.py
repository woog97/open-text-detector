import csv
import tempfile
import unittest
from pathlib import Path
from ingest_essay_sources import SOURCES, check_manifest, unique_train_rows, rejection, make_pair, balanced_rows

class EssaySourceTests(unittest.TestCase):
    def row(self, text=None):
        return {'essay_id':'essay1','prompt_name':'Phones and driving','competition_set':'train',
                'full_text':text or 'The driver should pay attention to the road and the people in it. '*20,
                'holistic_essay_score':'3','_source_row':0,'_annotation_count':1}
    def manifest(self):
        return {'source_id':'persuade',**SOURCES['persuade'],'source_file':SOURCES['persuade']['filename'],
                'split':'train','retrieved_at':'2026-10-02'}
    def test_known_prompt_and_text_exclusions_and_split(self):
        row=self.row()
        self.assertEqual(rejection(row,'persuade',{'prompt_names':[' PHONES  AND DRIVING ']}),'protected_or_other_source_prompt_family')
        self.assertEqual(rejection({**row,'competition_set':'test'},'persuade',{}),'not_train')
        self.assertIsNone(rejection(row,'persuade',{}))
    def test_annotations_are_one_document_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.csv';r=self.row();columns=list(r)
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
                w.writerows([r,r,{**r,'essay_id':'test','competition_set':'test'},
                             {**r,'essay_id':'bad'},{**r,'essay_id':'bad','full_text':'conflict'}])
            rows,bad=unique_train_rows(p,'persuade')
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['_annotation_count'],2);self.assertEqual(bad,1)
    def test_unchanged_whole_essay_offsets_source_and_rights(self):
        row=self.row();record,raw=make_pair(row,'persuade',self.manifest())
        self.assertEqual(record['text'],raw['record']['text'][record['raw_start']:record['raw_end']])
        self.assertEqual(record['category'],'essays');self.assertFalse(record['training_eligible'])
        self.assertIn('by-nc-sa',record['license_evidence']);self.assertEqual(record['word_count'],len(row['full_text'].split()))
        self.assertIsNone(make_pair({**row,'competition_set':'test'},'persuade',self.manifest()))
    def test_manifest_pins_release_and_training_file(self):
        check_manifest('persuade',self.manifest())
        for k in ('revision','sha256','split','source_file'):
            with self.assertRaises(ValueError):check_manifest('persuade',{**self.manifest(),k:'wrong'})
    def test_balancing_is_deterministic_across_score_cells(self):
        rows=[{**self.row(),'essay_id':str(i),'holistic_essay_score':str(i%2)} for i in range(6)]
        a=list(balanced_rows(rows,'persuade'));b=list(balanced_rows(list(reversed(rows)),'persuade'))
        self.assertEqual([r['essay_id'] for r in a],[r['essay_id'] for r in b])
        self.assertEqual([r['holistic_essay_score'] for r in a],['0','1']*3)

if __name__=='__main__':unittest.main()
