import base64,copy,hashlib,json,unittest
from ingest_scp_cccc import eligible,pairs,Excluded,FOOTER
from scan_scp_cccc import slug
class Tests(unittest.TestCase):
 def fixture(self):
  text=('The old man walked through the quiet forest and he knew that the house was waiting for him. '*10+'\n\n')*20
  row={'id':'x','source':'cccc_CC-MAIN-2013-48','created':'2013-12-01T00:00:00Z','text':text,'metadata':{'warc_url':'http://www.scp-wiki.net/a-story','warc_date':'2014-01-03T00:00:00Z','content_type':'text/html'}};line=(json.dumps(row)+'\n').encode();w={'slug':'a-story','record':row,'upstream_json_line_base64':base64.b64encode(line).decode(),'upstream_json_line_sha256':hashlib.sha256(line).hexdigest(),'source_file':'shard.gz','source_row':1};index={'a-story':{'domain':'scp-wiki.wikidot.com','tags':['tale'],'page_id':12,'creator':'Writer','created_at':'2010-01-01T00:00:00','history':[{'author':'Writer','date':'2025-01-01T00:00:00'}]}};return w,index
 def test_identity_and_history_separation(self):
  w,i=self.fixture();self.assertIsNotNone(eligible(w,i));i['a-story']['created_at']='2020-01-01T00:00:00'
  with self.assertRaises(Excluded):eligible(w,i)
  self.assertIsNone(slug('https://evil.com/a-story'));self.assertIsNone(slug('http://www.scp-wiki.net/a-story?comments=1'))
 def test_historical_and_original_guards(self):
  w,i=self.fixture();w['record']['metadata']['warc_date']='2024-01-01T00:00:00Z'
  with self.assertRaises(Excluded):eligible(w,i)
  w,i=self.fixture();w['record']['text']+='tampered'
  with self.assertRaises(Excluded):eligible(w,i)
 def test_genre_and_exact_offsets(self):
  w,i=self.fixture();items=list(pairs(w,i,[10]*4));self.assertTrue(items)
  for row,raw in items:
   self.assertEqual(row['text'],raw['record']['text'][row['raw_start']:row['raw_end']]);self.assertFalse(row['training_eligible']);self.assertIn('current_pinned',row['author_attribution_basis'])
  w['record']['text']='Special Containment Procedures: '+w['record']['text']
  with self.assertRaises(Excluded):eligible(w,i)
if __name__=='__main__':unittest.main()
