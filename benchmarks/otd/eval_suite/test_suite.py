import gzip,json,tempfile,unittest
from pathlib import Path
from suite import validate,digest
class ValidationTests(unittest.TestCase):
 def fixture(self,folder):
  row={'id':'a','text':'Human. AI.','text_sha256':digest(b'Human. AI.'),'regions':[{'start':0,'end':7,'label':0},{'start':7,'end':10,'label':1}]}
  blob=(json.dumps(row)+'\n').encode();(folder/'test.jsonl.gz').write_bytes(gzip.compress(blob));(folder/'runner.py').write_text('frozen')
  manifest={'code':{'runner.py':digest(b'frozen')},'profiles':{'test':{'sha256':digest(blob),'rows':1}}};(folder/'manifest.json').write_text(json.dumps(manifest));return manifest,row
 def test_valid(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.fixture(p);validate(p)
 def test_code_drift(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.fixture(p);(p/'runner.py').write_text('changed')
   with self.assertRaisesRegex(ValueError,'Code drift'):validate(p)
 def test_data_drift(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.fixture(p);(p/'test.jsonl.gz').write_bytes(gzip.compress(b'{}'))
   with self.assertRaisesRegex(ValueError,'Data drift'):validate(p)
 def test_bad_provenance(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);m,r=self.fixture(p);r['regions'][1]['label']=2;blob=json.dumps(r).encode();(p/'test.jsonl.gz').write_bytes(gzip.compress(blob));m['profiles']['test']['sha256']=digest(blob);(p/'manifest.json').write_text(json.dumps(m))
   with self.assertRaisesRegex(ValueError,'Region'):validate(p)
if __name__=='__main__':unittest.main()
