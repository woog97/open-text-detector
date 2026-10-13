import json,unittest
from mirror_core import sha,validate_parent,writer_messages,parse_topic,assess

class MirrorTests(unittest.TestCase):
 def setUp(self):
  self.text=' '.join('uniqueword'+str(i) for i in range(100))
  self.row={'record_id':'fixture','text':self.text,'passage_sha256':sha(self.text),'category':'creative','admission_status':'quarantined_candidate','training_eligible':False}
 def test_quarantine_gate(self):
  with self.assertRaises(ValueError):validate_parent(self.row)
  validate_parent(self.row,pilot=True)
 def test_writer_never_gets_source_or_extra_metadata(self):
  self.row.update(title=self.text,notes=self.text)
  payload=json.loads(writer_messages(self.row,'A traveler finding a new home')[1]['content'])
  self.assertEqual(set(payload),{'topic','genre','language','target_words'})
  self.assertNotIn(self.text,json.dumps(payload))
 def test_topic_copy_rejected(self):
  with self.assertRaises(ValueError):parse_topic(json.dumps({'topic':' '.join(self.text.split()[:12])}),self.text)
 def test_verbatim_generation_rejected(self):
  result=assess(self.row,self.text,{'input_tokens':20,'output_tokens':130},'stop')
  self.assertIn('verbatim_copy',result['flags'])
 def test_truncation_rejected(self):
  different=' '.join('newword'+str(i) for i in range(100))
  result=assess(self.row,different,{'input_tokens':20,'output_tokens':130},'length')
  self.assertIn('incomplete_generation',result['flags'])
 def test_frozen_family_required(self):
  self.row.update(admission_status='admitted',training_eligible=True,protected_overlap_status='passed')
  with self.assertRaises(ValueError):validate_parent(self.row)
  self.row.update(document_family_id='family-1',split='train');validate_parent(self.row)
 def test_hash_mismatch(self):
  self.row['text']='changed'
  with self.assertRaises(ValueError):validate_parent(self.row,pilot=True)
 def test_novel_generation_passes_mechanical_checks(self):
  different=' '.join('newword'+str(i) for i in range(100))
  self.assertTrue(assess(self.row,different,{'input_tokens':20,'output_tokens':130},'stop')['passed'])
if __name__=='__main__':unittest.main()
