import unittest
import numpy as np
from common import labels_from_regions,window_starts,choose_threshold,counts

class SupervisionTests(unittest.TestCase):
    def test_unicode_boundaries_and_whitespace(self):
        text='αβ  AI'
        regions=[{'start':0,'end':4,'label':0},{'start':4,'end':6,'label':1}]
        self.assertEqual(labels_from_regions(text,[(0,2),(2,4),(3,5),(1,5),(0,0)],regions),[0,-100,1,-100,-100])
    def test_windows_cover_every_token(self):
        for n in [1,509,510,511,766,1000,1500]:
            covered=np.zeros(n,dtype=int)
            for s in window_starts(n):covered[s:s+510]+=1
            self.assertTrue(np.all(covered>0))
            self.assertTrue(all(s>=0 and s+min(n,510)<=n for s in window_starts(n)))
    def test_threshold_never_exceeds_fpr_even_with_ties(self):
        for n in [5,99,100,101,1000]:
            p=np.round(np.linspace(.1,.9,n),1).astype(np.float32);y=np.zeros(n,dtype=int)
            threshold=choose_threshold(p,y,.01)
            self.assertLessEqual(counts(p,y,threshold)[1]/n,.01)
    def test_no_human_calibration_rejected(self):
        with self.assertRaises(ValueError):choose_threshold([.5],[1])
if __name__=='__main__':unittest.main()
