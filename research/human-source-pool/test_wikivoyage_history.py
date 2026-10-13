import unittest
from ingest_wikivoyage_history import extract_wikitext,make_pair,title_key
class HistoryTests(unittest.TestCase):
 def test_no_current_template_or_media_content(self):
  text=extract_wikitext('Before {{listing|content=AUTOMATED TEMPLATE}} after. [[File:x.jpg|IMAGE]] <ref>CITATION</ref> [[Place|display]]')
  self.assertNotIn('AUTOMATED',text);self.assertNotIn('IMAGE',text);self.assertNotIn('CITATION',text);self.assertIn('display',text)
 def test_revision_cutoff_and_exact_extracted_offsets(self):
  wiki='\n'.join('The city offers visitors a place to learn about its history and the local people. You can travel through the area with a guide and find useful information for your journey. '+str(i)+'.' for i in range(60))
  value={'query':{'pages':[{'pageid':12,'ns':0,'title':'Test City','revisions':[{'revid':34,'timestamp':'2021-01-01T00:00:00Z','user':'Editor','slots':{'main':{'contentmodel':'wikitext','content':wiki}}}]}]}}
  row,raw=make_pair(value,{'path':'original.json','retrieved_at':'today'},1);self.assertEqual(row['length_bin'],3);self.assertEqual(raw['record']['text'][row['raw_start']:row['raw_end']],row['text']);self.assertFalse(row['training_eligible'])
  value['query']['pages'][0]['revisions'][0]['timestamp']='2022-01-01T00:00:00Z';self.assertIsNone(make_pair(value,{'path':'original.json','retrieved_at':'today'},1))
 def test_page_identity(self):self.assertEqual(title_key('New_York'),title_key('New York'))
if __name__=='__main__':unittest.main()
