import unittest
import numpy as np
from score_wide_eval import summarize_row,aggregate,auroc

class WiderEvaluationTests(unittest.TestCase):
 def row(self,**kw):return {'id':'a','dataset':'test','cohort':'c','group_id':'paper','text':'Human. AI.','text_sha256':'hash','label':'mixed','granularity':'publisher_character_spans',**kw}
 def test_known_spans_keep_mixed_boundary_mask(self):
  r=self.row(regions=[{'start':0,'end':7,'label':0},{'start':7,'end':10,'label':1}])
  s=summarize_row(r,[(0,6),(6,9),(9,10)],np.array([.1,.9,.9]),{'tokens':.5,'sentences':.5})
  self.assertEqual(s['token_counts'],[2,0,0,1]);self.assertEqual(s['sentence_counts'],[1,0,0,1])
 def test_document_labels_do_not_invent_token_ground_truth(self):
  r=self.row(label='human',granularity='document_native_label')
  s=summarize_row(r,[(0,6),(7,10)],np.array([.1,.9]),{'tokens':.5,'sentences':.5})
  self.assertIsNone(s['token_counts']);self.assertIsNone(s['sentence_counts'])
 def test_ambiguous_polish_excluded_from_pure_binary(self):
  r=self.row(label='human',granularity='assisted_or_ambiguous_native_label')
  s=summarize_row(r,[(0,6),(7,10)],np.array([.1,.9]),{'tokens':.5,'sentences':.5})
  self.assertEqual(aggregate([s])['document_native_label_metrics']['evaluated_documents'],0)
 def test_exact_duplicate_has_one_vote(self):
  r=self.row(label='ai',granularity='observed_generated_response')
  s=summarize_row(r,[(0,6),(7,10)],np.array([.9,.9]),{'tokens':.5,'sentences':.5})
  a=aggregate([s,{**s,'id':'b'}]);self.assertEqual(a['unique_texts'],1);self.assertEqual(a['document_native_label_metrics']['tp'],1)
 def test_conflicting_span_annotations_excluded(self):
  r=self.row(label='ai',granularity='observed_generated_response')
  s=summarize_row(r,[(0,6),(7,10)],np.array([.9,.9]),{'tokens':.5,'sentences':.5})
  s['span_annotation_conflict']=True
  result=aggregate([s]);self.assertIsNone(result['span_or_historical_human_token_metrics']);self.assertEqual(result['span_annotation_conflict_rows'],1)
 def test_auc_ties_and_order(self):
  self.assertEqual(auroc([.1,.8],[0,1]),1)
  self.assertEqual(auroc([.8,.1],[0,1]),0)
  self.assertEqual(auroc([.5,.5],[0,1]),.5)
if __name__=='__main__':unittest.main()
