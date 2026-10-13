"""Run on Space: test label handling without loading a model."""
import unittest,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'package'))
import numpy as np
from score_wide_eval import summarize_row,aggregate
class MetricsTests(unittest.TestCase):
 def row(self,**kw):
  d={'id':'x','dataset':'test','text':'Human AI','text_sha256':hashlib.sha256(b'Human AI').hexdigest(),'label':'ai','granularity':'document_native_label'};d.update(kw);return d
 def score(self,r):
  return summarize_row(r,[(0,5),(6,8)],np.array([.1,.9]),{'tokens':.5,'sentences':.5})
 def test_document_labels_do_not_invent_token_gold(self):
  s=self.score(self.row());self.assertIsNone(s['token_counts']);self.assertIsNone(s['sentence_counts'])
 def test_known_boundaries(self):
  s=self.score(self.row(regions=[{'start':0,'end':5,'label':0},{'start':6,'end':8,'label':1}]));self.assertEqual(sum(s['token_counts']),2);self.assertEqual(s['token_counts'],[1,0,0,1])
 def test_unknown_assistance_not_accuracy_gold(self):
  s=self.score(self.row(label='mixed',regions=[{'start':0,'end':8,'label':-100}]));a=aggregate([s],False);self.assertEqual(a['document_native_label_metrics']['evaluated_documents'],0);self.assertEqual(sum(s['token_counts']),0)
if __name__=='__main__':unittest.main()
