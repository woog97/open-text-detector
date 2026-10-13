import gzip,json,tempfile,unittest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from upload_pool import validate
from collect_pool import digest

class ReleaseValidationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
  self.release=self.base/'intake'/'release';(self.release/'data').mkdir(parents=True)
  rawdir=self.base/'intake'/'sources'/'fixture';rawdir.mkdir(parents=True)
  self.text='The history of this work is in the record. '*10
  with gzip.open(rawdir/'documents.jsonl.gz','wt') as f:
   f.write(json.dumps({'raw_text_sha256':digest(self.text),'record':{'text':self.text}})+'\n')
  self.row=dict(source_id='fixture',category='reference',admission_status='quarantined_candidate',training_eligible=False,record_id='one',text=self.text,
   raw_text_sha256=digest(self.text),raw_start=0,raw_end=len(self.text),passage_sha256=digest(self.text),word_count=len(self.text.split()),source_revision='a'*40,reason_codes=['review_pending'],source_file='fixture.json.gz')
  self.summary=dict(admitted_total=0,candidate_total=1,sources=[dict(source_id='fixture',category='reference',candidate_passages=1,planned_passages=1)])
  (self.release/'source-registry.json').write_text(json.dumps({'sources':[{'id':'fixture','category':'reference'}]}))
 def tearDown(self):self.tmp.cleanup()
 def write(self,rows):
  pq.write_table(pa.Table.from_pylist(rows),self.release/'data'/'fixture.parquet')
  (self.release/'collection-summary.json').write_text(json.dumps(self.summary))
 def test_valid_offsets(self):
  self.write([self.row]);self.assertEqual(validate(self.base)['candidate_total'],1)
 def test_bad_offsets_rejected(self):
  self.row['raw_start']=1;self.write([self.row])
  with self.assertRaises(AssertionError):validate(self.base)
 def test_wrong_source_category_rejected(self):
  self.row['category']='news';self.write([self.row])
  with self.assertRaisesRegex(AssertionError,'source category'):validate(self.base)
 def test_missing_category_rejected(self):
  del self.row['category'];self.write([self.row])
  with self.assertRaisesRegex(AssertionError,'source category'):validate(self.base)
 def test_training_label_rejected(self):
  self.row['training_eligible']=True;self.write([self.row])
  with self.assertRaises(AssertionError):validate(self.base)
 def test_duplicate_rejected(self):
  second={**self.row,'record_id':'two'};self.summary['candidate_total']=2;self.summary['sources'][0].update(candidate_passages=2,planned_passages=2);self.write([self.row,second])
  with self.assertRaises(AssertionError):validate(self.base)
if __name__=='__main__':unittest.main()
