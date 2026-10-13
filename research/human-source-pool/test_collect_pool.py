import unittest,gzip,json,tempfile
from pathlib import Path
from collect_pool import make_passages, date_year, OPEN_LICENSE, digest,package

class IntakeTests(unittest.TestCase):
 def test_offsets_lengths_and_no_overlaps(self):
  paragraph='The account of this work is based on the observations and the evidence in the record. '
  text='\n\n'.join(paragraph*(i+2) for i in range(25))
  left=[3,3,3,3]
  out=make_passages(text,'fixture',10,left)
  self.assertTrue(out)
  for a,b,n,bin_id in out:
   self.assertEqual(len(text[a:b].split()),n)
  for i,(a,b,_,_) in enumerate(out):
   for c,d,_,_ in out[i+1:]:self.assertFalse(a<d and b>c)
  self.assertEqual(12-sum(left),len(out))
  self.assertEqual(out,make_passages(text,'fixture',10,[3,3,3,3]))
 def test_restricted_license_not_accepted(self):
  self.assertIsNone(OPEN_LICENSE.search('https://creativecommons.org/licenses/by-nc/4.0/'))
  self.assertIsNone(OPEN_LICENSE.search('https://creativecommons.org/licenses/by-nd/4.0/'))
  self.assertTrue(OPEN_LICENSE.search('https://creativecommons.org/licenses/by-sa/4.0/'))
 def test_dates(self):
  self.assertEqual(date_year('09-30-2024'),2024)
  self.assertEqual(date_year('2003-8-18'),2003)
  self.assertIsNone(date_year(None))
 def test_no_quota_no_records(self):
  self.assertEqual(make_passages('The text of the record. '*50,'x',0,[1,1,1,1]),[])
 def test_optional_source_fields_survive_shared_parquet_schema(self):
  import pyarrow.parquet as pq
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);quotas=[];statuses=[]
   for sid,extra in [('one',{'prompt_family_id':'prompt'}),('two',{'corpus_release_year':2011})]:
    folder=root/'sources'/sid;folder.mkdir(parents=True)
    with gzip.open(folder/'passages.jsonl.gz','wt') as f:f.write(json.dumps({'text':sid,'source_id':sid,**extra})+'\n')
    quotas.append({'source_id':sid,'category':'creative','planned_passages':1,'rights_evidence_status':'fixture'})
    statuses.append({'source_id':sid})
   package(root,{'sources':[]},{'planned_total':2,'source_quotas':quotas},statuses)
   a=pq.read_table(root/'release/data/one.parquet');b=pq.read_table(root/'release/data/two.parquet')
   self.assertEqual(a.schema,b.schema)
   self.assertEqual(a['prompt_family_id'].to_pylist(),['prompt'])
   self.assertEqual(b['prompt_family_id'].to_pylist(),[None])
   self.assertEqual(b['corpus_release_year'].to_pylist(),[2011])

if __name__=='__main__':unittest.main()
