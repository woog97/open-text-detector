import unittest
from data import layout,token_labels,crop,encode_example
class LayoutTests(unittest.TestCase):
 def test_repeat2_masks_exact_first_copy(self):
  e=layout([7,8,9],[0,-100,1],'causal',2)
  self.assertEqual(e['ids'],[7,8,9,7,8,9]);self.assertEqual(e['labels'],[-100,-100,-100,0,-100,1]);self.assertEqual(e['source_positions'],[3,4,5]);self.assertEqual(e['last_position'],5)
 def test_stage1_single_copy(self):
  e=layout([7,8],[0,1],'causal',1);self.assertEqual(e['ids'],[7,8]);self.assertEqual(e['source_positions'],[0,1])
 def test_encoder_specials_never_supervised(self):
  e=layout([7,8],[0,1],'encoder',2,101,102);self.assertEqual(e['labels'],[-100,0,1,-100]);self.assertEqual(e['source_positions'],[1,2]);self.assertEqual(e['last_position'],2)
 def test_boundary_and_unknown_mask(self):
  text='human AI unsure';regions=[{'start':0,'end':6,'label':0},{'start':6,'end':9,'label':1},{'start':9,'end':15,'label':-100}]
  self.assertEqual(token_labels(text,[(0,5),(4,8),(6,8),(8,9),(9,15)],regions),[0,-100,1,-100,-100])
 def test_crop_preserves_offsets(self):
  r={'id':'x','paper_id':'p','kind':'paired','text':'human AI','regions':[{'start':0,'end':6,'label':0},{'start':6,'end':8,'label':1}],'target_start':6,'target_end':8}
  c=crop(r,3,8);self.assertEqual(c['text'],'an AI');self.assertEqual(c['target_start'],3);self.assertEqual(c['regions'],[{'start':0,'end':3,'label':0},{'start':3,'end':5,'label':1}])
 def test_repetition_precedes_padding(self):
  a=layout([1,2],[0,1],'causal',2);b=layout([3],[1],'causal',2)
  self.assertEqual(b['ids']+[0]*(len(a['ids'])-len(b['ids'])),[3,3,0,0]);self.assertEqual(b['source_positions'],[1])

class StoragePolicyTests(unittest.TestCase):
 def test_local_model_loading_is_rejected(self):
  import runtime
  from unittest.mock import patch
  from pathlib import Path
  with patch.object(runtime,'ROOT',Path('/tmp/local-training-copy')):
   with self.assertRaisesRegex(RuntimeError,'restricted'):runtime.require_space()

if __name__=='__main__':unittest.main()
