import json
import unittest
from ingest_scp import eligibility,pairs,select_spans

class ScpTests(unittest.TestCase):
 def row(self):
  return {'domain':'scp-wiki.wikidot.com','tags':['tale'],'page_id':'42','creator':'Author',
  'created_at':'2019-01-01T00:00:00','history':[{'date':'2019-01-01T00:00:00','author':'Author'}],
  'raw_source':'The story of a person in the quiet room and the light outside. '*60,
  'title':'A Tale','url':'https://scp-wiki.wikidot.com/a-tale','link':'a-tale','hubs':['canon-one']}
 def test_requires_last_revision_not_only_creation_date(self):
  r=self.row();self.assertIsNone(eligibility(r));r['history'].append({'date':'2022-01-01T00:00:00','author':'Editor'})
  self.assertEqual(eligibility(r),'post2021_revision')
 def test_missing_date_creator_and_non_tale_excluded(self):
  r=self.row()
  for updated in [{'history':[]},{'creator':''},{'tags':['scp']},{'domain':'scp-cn.wikidot.com'}]:self.assertIsNotNone(eligibility({**r,**updated}))
 def test_markup_and_containment_not_sampled(self):
  markup='[[include component:license-box]]\n\n'+'Special Containment Procedures: '+'the person and the room in this place. '*30
  self.assertEqual(select_spans(markup,'id',[1,1,1,1]),[])
 def test_original_offsets_attribution_and_canon(self):
  r=self.row();results=list(pairs(r,'content_2019.json',{'retrieved_at':'2026-10-02'},[1,1,1,1],17));self.assertTrue(results)
  for row,raw in results:
   self.assertEqual(r['raw_source'][row['raw_start']:row['raw_end']],row['text']);self.assertEqual(row['source_row'],17)
   self.assertEqual(row['source_id'],'scp');self.assertEqual(row['category'],'creative');self.assertFalse(row['training_eligible'])
   self.assertEqual(json.loads(row['author_attribution_json'])[0]['name'],'Author');self.assertEqual(json.loads(row['canon_hub_ids_json']),['canon-one'])
   self.assertEqual(raw['record']['text'],r['raw_source']);self.assertIn('by-sa/3.0',row['license_evidence'])

if __name__=='__main__':unittest.main()
