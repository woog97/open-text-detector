import unittest
from enrich_paper_metadata import normalize, identities
from publish_browse_snapshot import build_rows
class MetadataTests(unittest.TestCase):
 def test_public_fields_only_and_legacy_normalization(self):
  note={'id':'forum','readers':['everyone'],'content':{'TL;DR':' Short summary ','keywords':'Learning; Robotics','primary_area':{'value':'Control'},'abstract':{'value':'secret','readers':['author']}}}
  row=normalize(note)
  self.assertEqual(row['tldr'],'Short summary');self.assertEqual(row['keywords'],['Learning','Robotics','Control']);self.assertEqual(row['detail']['abstract'],'')
  note['readers']=['author'];self.assertIsNone(normalize(note))
 def test_area_commas_remain_one_ground_truth_label(self):
  area='applications to physical sciences (physics, chemistry, biology, etc.)'
  row=normalize({'id':'forum','readers':['everyone'],'content':{'primary_area':{'value':area},'TLDR':{'value':'   '}}})
  self.assertEqual(row['primary_area'],area);self.assertEqual(row['keywords'],[area]);self.assertEqual(row['tldr'],'')
 def test_renamed_files_require_a_unique_title_in_the_same_collection(self):
  papers=[{'id':'pdf','filename':'Renamed.pdf','title':'A: Paper!','collection':'iclr/2025'}]
  source={'forum':{'id':'forum','title':'A Paper','conference':'iclr','year':2025}}
  self.assertEqual(identities(papers,source)['pdf'],'forum')
  source['duplicate']={**source['forum'],'id':'duplicate'}
  self.assertEqual(identities(papers,source)['pdf'],'Renamed')
 def test_enrichment_preserves_collections_and_scores(self):
  a='a'*64;b='b'*64
  rows=build_rows({f'papers/{x}.pdf':{'size':10} for x in [a,b]}, {'items':[{'id':'c'*24,'pdf_key':f'papers/{a}.pdf','classified':True,'models':['v8'],'collection':'colm/2024'}]}, {'papers':[]}, {'papers':[{'id':b[:24],'tldr':'new'}]}, {'papers':[{'id':a[:24],'collection':'colm/2024','tldr':'old','keywords':['test']}]})
  self.assertEqual([p['collection'] for p in rows],['colm/2024','iclr/2027']);self.assertTrue(rows[0]['classified']);self.assertEqual(rows[0]['models'],['v8']);self.assertEqual(rows[0]['tldr'],'old');self.assertEqual(rows[0]['id'],'c'*24)
if __name__=='__main__':unittest.main()
