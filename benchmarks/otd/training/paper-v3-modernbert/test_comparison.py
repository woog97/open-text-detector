import unittest
import numpy as np
from compare_models import project,units
from common import choose_threshold,counts
class ComparisonTests(unittest.TestCase):
 def test_projection_identical(self):
  np.testing.assert_array_equal(project('abcd',[(0,2),(2,4)],[(0,2),(2,4)],[1,3]),[1,3])
 def test_projection_overlap(self):
  np.testing.assert_allclose(project('abcdef',[(0,3),(3,6)],[(0,2),(2,5),(5,6)],[1,4,10]),[2,6])
 def test_projection_missing_text_rejected(self):
  with self.assertRaises(ValueError):project('ab',[(0,2)],[],[])
 def test_masked_target_boundary(self):
  r={'text':'ab cd.','regions':[{'start':0,'end':3,'label':-100},{'start':3,'end':6,'label':1}]}
  scores,labels,_,_=units(r,[(0,1),(1,4),(4,6)],np.array([.1,.2,.9]));np.testing.assert_array_equal(labels,[1]);np.testing.assert_array_equal(scores,[.9])
 def test_validation_threshold_ties(self):
  t=choose_threshold([1]*100,[0]*100);self.assertEqual(counts([1]*100,[0]*100,t)[1],0)
if __name__=='__main__':unittest.main()
