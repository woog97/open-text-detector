import unittest
from data import encode_example
class Tokenizer:
 cls_token_id=1;sep_token_id=2
 def __call__(self,text,**kw):
  return {'input_ids':list(range(10,len(text)+10)),'offset_mapping':[(i,i+1) for i in range(len(text))]}
class EncodingTests(unittest.TestCase):
 def test_long_document_keeps_all_content_and_no_local_labels(self):
  r={'text':'x'*1200,'supervision':'document_only','document_label':1,'token_labels':None}
  e=encode_example(r,Tokenizer(),'encoder',2)
  self.assertEqual([len(c['source_labels']) for c in e['chunks']],[510,510,180])
  for c in e['chunks']:
   self.assertEqual(set(c['source_labels']),{-100});self.assertEqual(c['sentence_groups'],[])
   self.assertEqual(c['segment_label'],-100);self.assertEqual(c['mixed_label'],-100)
 def test_document_cannot_claim_token_gold(self):
  with self.assertRaises(ValueError):encode_example({'text':'x','supervision':'document_only','document_label':1,'regions':[{'start':0,'end':1,'label':1}]},Tokenizer(),'encoder',2)
 def test_boundary_token_is_ignored(self):
  from data import token_labels
  self.assertEqual(token_labels('abcd',[(0,1),(1,3),(3,4)],[{'start':0,'end':2,'label':0},{'start':2,'end':4,'label':1}]),[0,-100,1])
if __name__=='__main__':unittest.main()
